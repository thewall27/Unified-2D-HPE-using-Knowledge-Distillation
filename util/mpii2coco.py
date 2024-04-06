from PIL import Image
import os.path as osp
import numpy as np
import json


db_type = 'val' # 'train'
mpii_file_path = f'data/mpii/annotations/mpii_{db_type}.json' 
save_path = 'util/mpii2coco_' + db_type + '.json'

f = open(mpii_file_path)
annot_file = json.load(f)

joint_num = 16
img_num = len(annot_file)

aid = 0
coco = {'images': [], 'categories': [], 'annotations': []}

category = {
    "supercategory": "person",
    "id": 1,  # to be same as COCO
    "name": "person",
    "skeleton": [[0,1],
        [1,2], 
        [2,6], 
        [7,12], 
        [12,11], 
        [11,10], 
        [5,4], 
        [4,3], 
        [3,6], 
        [7,13], 
        [13,14], 
        [14,15], 
        [6,7], 
        [7,8], 
        [8,9]] ,
    "keypoints": ["r_ankle", "r_knee","r_hip", 
                    "l_hip", "l_knee", "l_ankle",
                  "pelvis", "throax",
                  "upper_neck", "head_top",
                  "r_wrist", "r_elbow", "r_shoulder",
                  "l_shoulder", "l_elbow", "l_wrist"]}

coco['categories'] = [category]

for img_id in range(img_num):
    filename = str(annot_file[img_id]['image'])
    img = Image.open(osp.join('data/mpii/images', filename))
    w,h = img.size
    img_dict = {'id': img_id, 'file_name': filename, 'width': w, 'height': h}
    coco['images'].append(img_dict)

    kps = np.zeros((joint_num,3)) # xcoord, ycoord, vis
    joint_vis = annot_file[img_id]['joints_vis']

    for i in range(len(joint_vis)):
        if joint_vis[i] == 1:
            kps[i][0] = annot_file[img_id]['joints'][i][0]
            kps[i][1] = annot_file[img_id]['joints'][i][1]
            kps[i][2] = 1
    
    center = annot_file[img_id]['center']
    scale = annot_file[img_id]['scale']

    person_dict = {'id': aid, 'image_id': img_id, 'category_id': 1, 'iscrowd': 0, 'keypoints': kps.reshape(-1).tolist(),
                    'num_keypoints': int(np.sum(kps[:,2]==1)), 'center': center, 'scale': scale}
    coco['annotations'].append(person_dict)
    aid += 1

with open(save_path, 'w') as f:
    json.dump(coco, f)



