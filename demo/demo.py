# from mmpose.apis import inference_topdown, init_model
# from mmpose.utils import register_all_modules

# register_all_modules()

# config_file = 'demo/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
# checkpoint_file = 'demo/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
# model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cpu'

# # please prepare an image with person
# results = inference_topdown(model, 'demo/demo.jpg')
# print(results)


# build the inferencer with model config name
from mmpose.apis import MMPoseInferencer

img_path = 'demo/demo.jpg'
inferencer = MMPoseInferencer('rtmpose-m_8xb256-420e_body8-256x192')
result_generator = inferencer(img_path, show=False)
result = next(result_generator)
print(result)