import sys
sys.path.append(r'./src')
from data.coco_dataloader import DataSetCoco
from utils.img_utils import draw_single_bbox_annotation
from config.config import TaskConfig
import torch
import os
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # only_vis_class_list = ["cat", "dog"]
    # only_vis_class_list = ["person"]
    show = False
    config = TaskConfig()
    dataset = DataSetCoco("val", config)

    # ---------------------------------  Get Class Info ---------------------------------#
    # Print class index based on class text name
    # class_name = 'cat'
    class_name = 'liangtong'
    ids = dataset.coco.getCatIds(class_name)
    print("{}:{}".format(class_name, ids))
    # Print class detailed info based on class index
    cats = dataset.coco.loadCats(ids)
    print("{}:{}".format(ids, cats))
    print("#"*30)

    # ---------------------------------- Get Images ID  ---------------------------------#
    # Get Specific class index related images
    id = 17
    imgIds = dataset.coco.getImgIds(catIds=[id])
    print(imgIds)
    # Get Specific class name related images
    class_name = 'cat'
    ids = dataset.coco.getCatIds(class_name)
    imgIds = dataset.coco.getImgIds(catIds=[id])
    print(imgIds)
    print("#"*30)
    # ---------------------------------  Get Images INFO  -------------------------------#
    imgInfo = dataset.coco.loadImgs(imgIds)[0]
    print(imgInfo)
    # ---------------------------------  Show Annotation Bbox -------------------------------#
    cocoRoot = "/data/yyh/2020/CenterNet/data/coco"
    dataType = "val2017"
    for img_id in imgIds:
        # img_id = "484279"
        print(img_id)
        show = False
        imgInfo = dataset.coco.loadImgs(img_id)[0]
        imPath = os.path.join(cocoRoot, 'images', dataType, imgInfo['file_name'])
        print(imPath)
        print("#"*30)
        annIds = dataset.coco.getAnnIds(imgIds=imgInfo['id'])
        im = cv2.imread(imPath)
        anns = dataset.coco.loadAnns(annIds)
        for single_boox in anns:
            label_name = dataset.coco.loadCats([single_boox["category_id"]])[0]["name"]
            if label_name in only_vis_class_list:
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
    

    # # ---------------------------------  Show Annotation Mask  -----------------------------#
    # # plt.imshow(im)
    # imgInfo = dataset.coco.loadImgs(458255)[0]
    # annIds = dataset.coco.getAnnIds(imgIds=imgInfo['id'])
    # anns = dataset.coco.loadAnns(annIds)
    # mask = dataset.coco.annToMask(anns[3])
    # dataset.coco.showAnns(anns)
    # # plt.imshow(mask)