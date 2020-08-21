import sys
sys.path.append(r'./src')
import cv2
import os
import torch
from config.config import TaskConfig
from utils.img_utils import draw_bbox_annotation
from data.coco_dataloader import DataSetCoco
import random
import albumentations as A
import copy


if __name__ == "__main__":
    # show all dataset
    transform = A.Compose(
        [ 
            # A.RandomCropNearBBox(p=1),
            # Dropout
            A.Cutout(p=0.3),
            
            # Color
            A.RandomBrightnessContrast(p=0.5),
            A.CLAHE(p=0.1),
            A.RGBShift(p=0.1),

            # Blur
            A.MotionBlur(p=0.2),

            # Noise
            A.GaussNoise(p=0.2),

            # Spatial
            A.Flip(p=0.5),                                                 # either horizontally, vertically or both horizontally and vertically
            A.ShiftScaleRotate(scale_limit=(-0.3, 0.3), shift_limit=(-0.3, 0.3), rotate_limit=(0, 0), border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2),
            A.RandomCrop(height=480, width=640, p=0.2),
            # A.RandomRotate90()
            # A.RandomScale(always_apply=True, scale_limit=(0.5, 2.0))
            A.Resize(height=480, width=640)
        ],
        bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['category_ids']),
    )
    random.seed(7)
    cocoRoot = "/data/yyh/2020/Simple-Centernet-with-Visualization-Pytorch_/dataset/code_testing_coco"
    dataType = "val2017"
    config = TaskConfig()
    dataset = DataSetCoco("val", config)

    for img_id in dataset.images:
        imgInfo = dataset.coco.loadImgs(img_id)[0]
        imPath = os.path.join(cocoRoot, dataType, imgInfo['file_name'])
        print(imPath)
        annIds = dataset.coco.getAnnIds(imgIds=imgInfo['id'])
        im = cv2.imread(imPath)
        origin_img = copy.deepcopy(im)
        anns = dataset.coco.loadAnns(annIds)
        bbox_list = []
        label_name_list = []
        for single_boox in anns:
            label_name = dataset.coco.loadCats([single_boox["category_id"]])[0]["name"]
            label_name_list.append(label_name)
            x_min, y_min, width, height = single_boox["bbox"]
            x_max = x_min + width
            y_max = y_min + height
            center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
            bbox = [x_min, y_min, x_max, y_max]
            bbox_list.append(bbox)
            # draw original bbox
            origin_img = draw_bbox_annotation(origin_img, bbox, center, label_name)

        # cv2.imshow("origin_im", origin_img)
     
        # aug
        transformed = transform(image=im, bboxes=bbox_list, category_ids=label_name_list)
        im = transformed['image']
        # print(transformed["bboxes"])
        print("#"*30)
        for single_bbox, label_name in zip(transformed["bboxes"], transformed['category_ids']):
            x_min, y_min, x_max, y_max = single_bbox
            center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
            # draw aug bbox
            im = draw_bbox_annotation(im, single_bbox, center, label_name)
        cv2.imshow("aug", im)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
            cv2.destroyAllWindows()
