import sys
sys.path.append(r'./src')
from data.coco_dataloader import DataSetCoco
from utils.img_utils import draw_single_bbox_annotation
from config.config import TaskConfig
import torch
import os
import cv2
import matplotlib.pyplot as plt
import pycocotools.coco as coco

if __name__ == "__main__":
    # only_vis_class_list = ["cat", "dog"]
    only_vis_class_list = ["person"]
    show = False
    config = TaskConfig()
    coco_ = coco.COCO("/data/yyh/2020/CenterNet/data/coco/annotations/instances_val2017.json")

    # ---------------------------------  Get Class Info ---------------------------------#
    # Print class index based on class text name
    # class_name = 'cat'
    class_name = 'person'
    ids = coco_.getCatIds(class_name)
    print("{}:{}".format(class_name, ids))
    # Print class detailed info based on class index
    cats = coco_.loadCats(ids)
    print("{}:{}".format(ids, cats))
    print("#"*30)

    # ---------------------------------- Get Images ID  ---------------------------------#
    # Get Specific class index related images
    id = 17
    imgIds = coco_.getImgIds(catIds=[id])
    print(imgIds)
    # Get Specific class name related images
    class_name = 'cat'
    ids = coco_.getCatIds(class_name)
    imgIds = coco_.getImgIds(catIds=[id])
    print(imgIds)
    print("#"*30)
    # ---------------------------------  Get Images INFO  -------------------------------#
    imgInfo = coco_.loadImgs(imgIds)[0]
    print(imgInfo)
    # ---------------------------------  Show Annotation Bbox -------------------------------#
    cocoRoot = "/data/yyh/2020/CenterNet/data/coco"
    dataType = "val2017"
    for img_id in imgIds:
        # img_id = "484279"
        print(img_id)
        show = False
        imgInfo = coco_.loadImgs(img_id)[0]
        imPath = os.path.join(cocoRoot, dataType, imgInfo['file_name'])
        print(imPath)
        print("#"*30)
        annIds = coco_.getAnnIds(imgIds=imgInfo['id'])
        im = cv2.imread(imPath)
        anns = coco_.loadAnns(annIds)
        for single_boox in anns:
            label_name = coco_.loadCats([single_boox["category_id"]])[0]["name"]
            if label_name in only_vis_class_list:
                show = True
                x_min, y_min, width, height = single_boox["bbox"]
                center = [x_min + width / 2, y_min + height / 2]  
                draw_single_bbox_annotation(im, single_boox["bbox"], label_name, center, type="coco")
        if show:
            cv2.imshow("im", im)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
                cv2.destroyAllWindows()
    

    # # ---------------------------------  Show Annotation Mask  -----------------------------#
    # # plt.imshow(im)
    # imgInfo = coco_.loadImgs(458255)[0]
    # annIds = coco_.getAnnIds(imgIds=imgInfo['id'])
    # anns = coco_.loadAnns(annIds)
    # mask = coco_.annToMask(anns[3])
    # coco_.showAnns(anns)
    # # plt.imshow(mask)