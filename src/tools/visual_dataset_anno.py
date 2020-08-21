import sys
sys.path.append(r'./src')
import cv2
import os
import torch
from config.config import TaskConfig
from utils.img_utils import draw_bbox_annotation
from data.coco_dataloader import DataSetCoco



if __name__ == "__main__":
    # show all dataset
    cocoRoot = "/data/yyh/2020/Simple-Centernet-with-Visualization-Pytorch_/dataset/code_testing_coco"
    dataType = "val2017"
    show_only = False
    only_vis_class_list = ["cat", "dog"]
    show = False
    config = TaskConfig()
    dataset = DataSetCoco("val", config)

    for img_id in dataset.images:
        show = False
        imgInfo = dataset.coco.loadImgs(img_id)[0]
        imPath = os.path.join(cocoRoot, dataType, imgInfo['file_name'])
        print(imPath)
        annIds = dataset.coco.getAnnIds(imgIds=imgInfo['id'])
        im = cv2.imread(imPath)
        anns = dataset.coco.loadAnns(annIds)
        for single_boox in anns:
            label_name = dataset.coco.loadCats(
                [single_boox["category_id"]])[0]["name"]
            if show_only:
                if label_name in only_vis_class_list:
                    show = True
                    x_min, y_min, width, height = single_boox["bbox"]
                    x_max = x_min + width
                    y_max = y_min + height
                    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                    bbox = [x_min, y_min, x_max, y_max]
                    draw_bbox_annotation(im, bbox, center, label_name)
            else:
                show = True
                x_min, y_min, width, height = single_boox["bbox"]
                x_max = x_min + width
                y_max = y_min + height
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                bbox = [x_min, y_min, x_max, y_max]
                draw_bbox_annotation(im, bbox, center, label_name)

        if show:
            cv2.imshow("im", im)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
                cv2.destroyAllWindows()
