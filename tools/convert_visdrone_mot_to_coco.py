import os
from unicodedata import category
import numpy as np
import json
import cv2

from yolox import data


DATA_PATH = os.path.join('..', 'datasets', 'visdrone')
OUT_PATH = OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['VisDrone2019-MOT-test-dev'] #'VisDrone2019-MOT-train', 'VisDrone2019-MOT-test-dev', 'VisDrone2019-MOT-val']
CATEGORIES = [{
          "id": 1,
          "name": "pedestrian",
          "supercategory": "none"},
          {
          "id": 2,
          "name": "people",
          "supercategory": "none"},
          {
          "id": 3,
          "name": "bicycle",
          "supercategory": "none"},
          {
          "id": 4,
          "name": "car",
          "supercategory": "none"},
          {
          "id": 5,
          "name": "van",
          "supercategory": "none"},
          {
          "id": 6,
          "name": "truck",
          "supercategory": "none"},
          {
          "id": 7,
          "name": "tricycle",
          "supercategory": "none"},
          {
          "id": 8,
          "name": "awning-tricycle",
          "supercategory": "none"},
          {
          "id": 9,
          "name": "bus",
          "supercategory": "none"},
          {
          "id": 10,
          "name": "motor",
          "supercategory": "none"},
          {
          "id": 11,
          "name": "others",
          "supercategory": "none"}
          ]

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, split, 'sequences')
        out_path = os.path.join(OUT_PATH, f'{split}.json')
        out = {'images': [], 'annotations' : [], 'videos': [],
               'categories': CATEGORIES}
        print(os.getcwd())
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        for seq in sorted(seqs):
            video_cnt += 1
            out['videos'].append({'id' : video_cnt, 'file_name' : seq})
            seq_path = os.path.join(data_path, seq)
            img_path = seq_path
            ann_path = os.path.join(DATA_PATH, split, 'annotations', os.path.basename(seq_path) + '.txt')
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])
            image_range = [0, num_images-1]
            for i in range(num_images):
                img = cv2.imread(os.path.join(img_path, file_name := f'{i+1:07d}.jpg'))
                height, width = img.shape[:2]
                image_info = {'file_name' : os.path.join("sequences", seq, file_name),
                              'id' : image_cnt + i + 1, # image n in the entire training set
                              'frame_id' : i + 1 - image_range[0], # image n in the video seq, starts in 1
                              'prev_image_id' : image_cnt + i if i > 0 else -1, # prev img n in the entire set
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id' : video_cnt,
                              'height' : height, 'width': width
                }
                out['images'].append(image_info)
            print(f'{seq} : {num_images} images')
            if split != 'VisDrone2019-MOT-test':
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                print(f'{int(anns[:, 0].max())} annotated images')
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    category_id = int(anns[i][7])
                    ann_cnt += 1
                    if category_id == 0:
                        continue # ignored region
                    # Maybe group all vehicles in one category
                    if not track_id == tid_last:
                                tid_curr += 1
                                tid_last = track_id
                    ann = {'id': ann_cnt,
                           'category_id' : category_id,
                           'image_id' : image_cnt + frame_id,
                           'track_id' : tid_curr,
                           'bbox' : anns[i][2:6].tolist(),
                           'conf' : float(anns[i][6]),
                           'iscrowd' : 0,
                           'area' : float(anns[i][4] * anns[i][5])
                    }
                    out['annotations'].append(ann)
            image_cnt += num_images
            print(tid_curr, tid_last)
        print(f"loaded {split} for {len(out['images'])}, and {len(out['annotations'])} samples")
        json.dump(out, open(out_path, 'w'))





