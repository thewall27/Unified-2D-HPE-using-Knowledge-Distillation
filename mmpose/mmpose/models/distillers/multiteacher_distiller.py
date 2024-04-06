from abc import ABCMeta
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.logging import MessageHub
from mmengine.model import BaseModel
from mmengine.runner.checkpoint import load_checkpoint
from torch import Tensor

from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models import build_pose_estimator
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ForwardResults, OptConfigType, OptMultiConfig,
                                 OptSampleList, SampleList)


@MODELS.register_module()
class MultiTeacherDistiller(BaseModel, metaclass=ABCMeta):
    """This distiller is designed for distillation of RTMPose.

    It typically consists of teacher_model and student_model. Please use the
    script `tools/misc/pth_transfer.py` to transfer the distilled model to the
    original RTMPose model.

    Args:
        teacher1_cfg (str): Config file of the teacher #1 model.
        teacher2_cfg (str): Config file of the teacher #2 model.
        student_cfg (str): Config file of the student model.
        distill_cfg (dict): Config for distillation. Defaults to None.
        weight(list):[student_weight, coco_teacher_weight, mpii_teacher_weight]
        teacher1_pretrained (str): Path of the pretrained teacher #1 model.
            Defaults to None.
        teacher2_pretrained (str): Path of the pretrained teacher #2 model.
            Defaults to None.
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
    """

    def __init__(self,
                 teacher1_cfg,
                 teacher2_cfg,
                 student_cfg,
                 distill_cfg=None,
                 weight=None,
                 teacher1_pretrained=None,
                 teacher2_pretrained=None,
                 train_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.teacher1 = build_pose_estimator(
            (Config.fromfile(teacher1_cfg)).model)
        self.teacher2 = build_pose_estimator(
            (Config.fromfile(teacher2_cfg)).model)
        self.teacher1_pretrained = teacher1_pretrained
        self.teacher2_pretrained = teacher2_pretrained
        self.teacher1.eval()
        for param in self.teacher1.parameters():
            param.requires_grad = False
        self.teacher2.eval()
        for param in self.teacher2.parameters():
            param.requires_grad = False

        self.student = build_pose_estimator(
            (Config.fromfile(student_cfg)).model)

        self.distill_cfg = distill_cfg
        self.distill_losses = nn.ModuleDict()
        self.weight = weight
        if self.distill_cfg is not None:
            for item_loc in distill_cfg:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    use_this = item_loss.use_this
                    if use_this:
                        self.distill_losses[loss_name] = (MODELS.build(item_loss))


        # self.two_dis = two_dis
        self.train_cfg = train_cfg if train_cfg else self.student.train_cfg
        self.test_cfg = self.student.test_cfg
        self.metainfo = self.student.metainfo

    def init_weights(self):
        if self.teacher1_pretrained is not None:
            load_checkpoint(
                self.teacher1, self.teacher1_pretrained, map_location='cpu')
        if self.teacher2_pretrained is not None:
            load_checkpoint(
                self.teacher2, self.teacher2_pretrained, map_location='cpu')
        self.student.init_weights()

    def set_epoch(self):
        """Set epoch for distiller.

        Used for the decay of distillation loss.
        """
        self.message_hub = MessageHub.get_current_instance()
        self.epoch = self.message_hub.get_info('epoch')
        self.max_epochs = self.message_hub.get_info('max_epochs')

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # use customed metainfo to override the default metainfo
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        self.set_epoch()

        losses = dict()

        with torch.no_grad():
            fea_t1 = self.teacher1.extract_feat(inputs)
            lt1_x, lt1_y = self.teacher1.head(fea_t1)
            pred_t1 = (lt1_x, lt1_y)

            fea_t2 = self.teacher2.extract_feat(F.interpolate(inputs, size=(256, 256), mode='bilinear', align_corners=True))
            lt2_x, lt2_y = self.teacher2.head(fea_t2)
            # Replace this with 1d interpolation with 512 -> 384
            lt2_x = F.interpolate(lt2_x, size=384, mode='linear', align_corners=True)
            pred_t2 = (lt2_x, lt2_y) # (B, 16, 512), (B, 16, 512) 

        # KLDiscretLoss
        fea_s = self.student.extract_feat(inputs)
        ori_loss, pred, gt, target_weight = self.head_loss(
            fea_s, data_samples, train_cfg=self.train_cfg)
        losses.update(ori_loss)

        all_keys = self.distill_losses.keys()

        # MPII 16 --> student 21 mapping
        mapping=[
            (0, 16),
            (1, 14),
            (2, 12),
            (3, 11),
            (4, 13),
            (5, 15),
            (6, 17),
            (7, 18),
            (8, 19),
            (9, 20),
            (10, 10),
            (11, 8),
            (12, 6),
            (13, 5),
            (14, 7),
            (15, 9),
        ]

        # KDLoss
        if 'loss_logit_coco' in all_keys:
            loss_name = 'loss_logit_coco'
            # Map keypoints from student 21 --> COCO 17 in pred_coco variable
            stud_x = pred[0]
            stud_y = pred[1]
            stud_coco_x = stud_x[:, :17]
            stud_coco_y = stud_y[:, :17]
            pred_coco = (stud_coco_x, stud_coco_y)

            coco_teacher_weight = self.weight[1]
            losses[loss_name] = coco_teacher_weight * self.distill_losses[loss_name](
                pred_coco, pred_t1, self.student.head.loss_module.beta,
                target_weight)

        if 'loss_logit_mpii' in all_keys:
            loss_name = 'loss_logit_mpii'
            # Map keypoints from student 21 --> MPII 16 in pred_mpii variable
            stud_x = pred[0] # (B, 21, 2*w)
            stud_y = pred[1] # (B, 21, 2*h)
            target_index, source_index = zip(*mapping)
            stud_x[:, target_index, :] = stud_x[:, source_index, :]
            stud_y[:, target_index, :] = stud_y[:, source_index, :]
            stud_mpii_x = stud_x[:, :16]
            stud_mpii_y = stud_y[:, :16]

            pred_mpii = (stud_mpii_x, stud_mpii_y)

            mpii_teacher_weight = self.weight[2]
            losses[loss_name] = mpii_teacher_weight * self.distill_losses[loss_name](
                pred_mpii, pred_t2, self.student.head.loss_module.beta,
                target_weight)

        # NOTE: Do we want to eep this line?
        losses[loss_name] = (1 - self.epoch / self.max_epochs) * losses[loss_name]

        # Pass these weights from config 
        student_weight = self.weight[0]
        losses['loss_total'] = student_weight * ori_loss['loss_kpt'] + \
            losses.get('loss_logit_coco', 0) + \
            losses.get('loss_logit_mpii', 0)

        """
        student_kpts = student(input)  # 21 keypoints
        if len(gt_kpts) == 17:
            teacher_kpts = teacher_coco(input)
            student_kpts = student_kpts[coco_indices]  # 17 keypoints
        else:
            teacher_kpts = teacher_mpii(inputs)
            student_kpts = student_kpts[mpii_indices]  # 16 keypoints
        loss_kpts = KLDiscretLoss(student_kpts, gt_kpts)
        loss_kd = KDLoss(student_kpts, teacher_kpts)
        loss = alpha * loss_kd + (1-alpha) * loss_kpts
        Here, alpha is a hyperparameter with values in range [0,1]. For example, you can set it to 0.3.
        """

        return losses

    def predict(self, inputs, data_samples):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        return self.student.predict(inputs, data_samples)

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.teacher.extract_feat(inputs)
        return x

    def head_loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.student.head.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.student.head.loss_module(pred_simcc, gt_simcc,
                                             keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.student.head.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses, pred_simcc, gt_simcc, keypoint_weights

    def _forward(self, inputs: Tensor):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            Union[Tensor | Tuple[Tensor]]: forward output of the network.
        """
        return self.student._forward(inputs)