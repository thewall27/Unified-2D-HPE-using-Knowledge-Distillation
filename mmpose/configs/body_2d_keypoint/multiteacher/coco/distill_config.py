_base_ = [
    '../../rtmpose/coco/rtmpose-m_8xb256-420e_mpii-coco-256x192.py'
]

# config settings
logit = True

# train_cfg = dict(max_epochs=60, val_interval=10)

# method details
model = dict(
    _delete_=True,
    type='MultiTeacherDistiller',
    teacher1_cfg='mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py',
    teacher2_cfg='mmpose/configs/body_2d_keypoint/rtmpose/mpii/rtmpose-m_8xb64-210e_mpii-256x256.py',
    student_cfg='mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_mpii-coco-256x192.py',
    distill_cfg=[
        dict(methods=[
            dict(
                type='KDLoss',
                name='loss_logit_coco',
                use_this=logit
            ),
            dict(
                type='KDLoss',
                name='loss_logit_mpii',
                use_this=logit
            )
        ]),
    ],
    weight=[0.3, 0.35, 0.35], # [student_weight, coco_teacher_weight, mpii_teacher_weight]
    teacher1_pretrained='work_dirs/rtmpose-m_8xb256-420e_coco-256x192/best_coco_AP_epoch_100.pth',
    teacher2_pretrained='work_dirs/rtmpose-m_8xb64-210e_mpii-256x256/best_PCK_epoch_100.pth',
    # train_cfg=train_cfg,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
)

optim_wrapper = dict(clip_grad=dict(max_norm=1., norm_type=2))