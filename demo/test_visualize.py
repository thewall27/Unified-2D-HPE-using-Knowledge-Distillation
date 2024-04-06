# from mmpose.apis import MMPoseInferencer

# img_path = 'demo/demo.jpg'

# # build the inferencer with model config path and checkpoint path/URL
# inferencer = MMPoseInferencer(
#     pose2d='mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_mpii-coco-256x192.py',
#     pose2d_weights='work_dirs/rtmpose-m_8xb256-420e_mpii-coco-256x192/best_coco_AP_epoch_50.pth'
# )
# result_generator = inferencer(img_path, show=False)
# result = next(result_generator)
# print(result)

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_mpii-coco-256x192.py'
checkpoint_file = 'work_dirs/rtmpose-m_8xb256-420e_mpii-coco-256x192/best_coco_AP_epoch_50.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cpu'

# please prepare an image with person
results = inference_topdown(model, 'demo/demo.jpg')
print(results)