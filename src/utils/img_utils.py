import cv2
import os
import numpy as np
# from .pycocotools import mask as maskUtils

font = cv2.FONT_HERSHEY_SIMPLEX


def visdom_show_opencv(vis, img, title, win):
    """
    Use visdom to show opencv numpy BGR picture in browser
    :param path： must be string
    :return: numpy.ndarray(uint8)
    """
    vis.image(img.transpose(2, 0, 1)[
              ::-1, ...], win=win, opts={'title': title.format(img.shape[1], img.shape[0])})


def read_image(path):
    """
    Read Single Image
    :param path： must be string
    :return: numpy.ndarray(uint8)
    """
    # return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    return cv2.imread(path)


def choose_color(color):
    if color == "red":
        bgr = (0, 0, 255)
    elif color == "blue":
        bgr = (255, 0, 0)
    elif color == "green":
        bgr = (0, 255, 0)
    elif color == "purple":
        bgr = (240, 32, 160)
    elif color == "black":
        bgr = (0, 0, 0)
    else:
        raise Exception(
            "Choose from \"red\", \"blue\", \"green\, \"purple\, \"black\", Please!!!! ")
    return bgr


def draw_point_on_img(img, position, color="green", size=5):
    """
    Draw Points on single image
    :param position: must be [x, y] list format
    :return: numpy.ndarray(uint8)
    """
    bgr = choose_color(color)
    x, y = [int(_) for _ in position]
    img_show = cv2.circle(img, (x, y), size, bgr, 1)
    return img_show


def draw_bbox_on_img(img, bbox, color="red", type="coco", size=2):
    """
    Draw Bounding Boxes on single image
    :param bbox: must be [xmin, ymin, xmax, ymax] list format
    :param type: can be "coco"([x_min, y_min, width, height]) or "voc"([x_min, y_min, x_max, y_max])
    :return: numpy.ndarray(uint8)
    """
    bgr = choose_color(color)
    if type == "coco":
        xmin, ymin, width, height = [int(_) for _ in bbox]
        xmax, ymax = xmin + width, ymin + height
    elif type == "voc":
        xmin, ymin, xmax, ymax = [int(_) for _ in bbox]

    img_show = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bgr, size)
    return img_show


def write_text_on_img(img, contant, position, color="red"):
    """
    Write Bounding Boxes Label on single image
    :param contant: must be string format
    :return: numpy.ndarray(uint8)
    """
    bgr = choose_color(color)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    img = cv2.putText(img, str(contant), (int(x), int(y)), font, 1, bgr, 1)
    return img


def draw_single_img_bbox_annotation(img, bbox_list, label_list, point_list=[], type="coco"):
    for index, bbox in enumerate(bbox_list):
        label = label_list[index]
        if len(point_list) == 0:
            img = draw_single_bbox_annotation(img, bbox, label, type=type)
        else:
            point = point_list[index]
            img = draw_single_bbox_annotation(img, bbox, point, label, type)


def draw_single_bbox_annotation(img, bbox, label, point=[], type="coco"):
    """
    Draw An annotation on single image
    :param img: must be numpy.ndarray(uint8)
    :param bbox: can be [xmin, ymin, xmax, ymax](pascal_voc format) or [x_min, y_min, width, height](coco format) format
    :param point: must be [x, y] list format
    :param label: must be string format
    :param type: can be "coco"([x_min, y_min, width, height]) or "voc"([x_min, y_min, x_max, y_max])
    :return: numpy.ndarray(uint8)
    """

    img = draw_bbox_on_img(img, bbox, type=type)
    if len(point) != 0:
        img = draw_point_on_img(img, point)

    if bbox[1] - 5 < 5:
        class_label_posiiton = [bbox[0], bbox[1] + 20]
    else:
        class_label_posiiton = [bbox[0], bbox[1] - 5]
    img = write_text_on_img(img, label, class_label_posiiton)
    return img


def video_to_stream_save(video_path="", save_name_prefix="", save_root="", save_gap=12):
    videoCapture = cv2.VideoCapture(video_path)
    success = True
    count = 0
    save_count = 0
    while success:
        ret, frame = videoCapture.read()
        if ret:
            count += 1
            if count % save_gap == 0:
                save_count += 1
                save_name = "{}{}.jpg".format(
                    save_name_prefix, str(save_count))
                save_path = os.path.join(save_root, save_name)
                cv2.imwrite(save_path, frame)
        else:
            break


if __name__ == "__main__":
    # video_to_stream_save(video_path="/data/yyh/2020/Simple-Centernet-with-Visualization-Pytorch_/dataset/daoguan_jiechu/video/2020_08_06_18_12_26/top.avi",
    #                      save_name_prefix="meijiechu1_", save_root="/data/yyh/2020/Simple-Centernet-with-Visualization-Pytorch_/dataset/daoguan_jiechu/meijiechu")
    video_to_stream_save(video_path="/data/yyh/2020/Simple-Centernet-with-Visualization-Pytorch_/dataset/daoguan_jiechu/video/2020_08_07_15_06_41/top.avi",
                         save_name_prefix="jiechu4_", save_root="/data/yyh/2020/Simple-Centernet-with-Visualization-Pytorch_/dataset/daoguan_jiechu/jiechu")