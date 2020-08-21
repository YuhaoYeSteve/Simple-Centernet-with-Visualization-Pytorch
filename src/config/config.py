from utils.general_utils import get_time, check_path_without_delete, check_file_without_delete
from utils.init_visdom import init_visdom_
import albumentations as A
import cv2
import os


class Config:
    def __init__(self):

        # ---------------------------   Hyper-Parameter  ------------------------------#

        self.model_name = "efficientnet-b0"  # "resnet50"/ SeNet / "efficientnet-b0"
        self.train_epoch = 100

        self.best_acc = 0.0
        self.class_num = 0
        self.task_name = "cifar10"
        self.batch_size = 64
        self.num_workers = 8
        self.input_size = 224
        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]
        self.model_and_training_info_save_root = "./train_out/"  #
        self.mix_up_alpha = 0.2
        self.print_loss_interval = 100
        self.print_loss_remainder = 99
        self.dataLoader_num_worker = 32
        self.class_weight = []
        self.pretrain_model_path = ""
        self.traing_time = get_time()
        # -------------------------------   Switch  ----------------------------------#
        # if use visdom
        self.use_visdom = True

        # if use mix up
        self.use_mix_up = False

        # if use label_smoothing on loss
        self.use_label_smoothing = False

        # if use class specified pre-trained model
        self.load_dataset_specified_pre_train = False

        # if use Apex FP16 training
        self.use_apex_amp_mix_precision = True

        # if use cudnn Accleration
        self.use_cudnn_accelerate = True

        # if use random seed(better for doing experiment)
        self.set_seed = True

        # if use multiple GPU
        self.use_multi_gpu = True

        # if use warm-up leanring rate
        self.if_warmup = False

        # -------------------------------   Choice  ----------------------------------#

        # Choose optimizer
        self.which_optimizer = "adam"  # "adam" or "sgd"

        # Choose GPU number
        if self.use_multi_gpu:
            # self.gpu_num = [3, 4, 5, 6]  # Multiple GPU
            # self.gpu_num = [4, 5, 6, 7]  # Multiple GPU
            self.gpu_num = [1, 2, 3]  # Multiple GPU
        else:
            self.gpu_num = "1"  # Single GPU

        self.base_lr = 0.001
        self.lr_schedule = {  # 学习率调整策略
            10: self.base_lr * 0.1,
            50: self.base_lr * 0.01,
            80: self.base_lr * 0.001
        }


class TaskConfig(Config):
    def __init__(self):
        super(TaskConfig, self).__init__()
        # self.data_root = "/data/yyh/2020/Simple-Centernet-with-Visualization-Pytorch/dataset/wuzhifenli"
        self.data_root = "/data/yyh/2020/Simple-Centernet-with-Visualization-Pytorch_/dataset/code_testing_coco"
        # self.data_root = "/data/yyh/2020/CenterNet/data/coco/"
        self.training_name = "test"
        self.log_and_model_root = os.path.join(
            "./train_out", self.training_name, self.traing_time)

        self.log_file_path = os.path.join(self.log_and_model_root, "log.txt")

        check_path_without_delete(self.log_and_model_root)
        check_file_without_delete(self.log_file_path)

        self.transform = A.Compose(
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
                # either horizontally, vertically or both horizontally and vertically
                A.Flip(p=0.5),
                A.ShiftScaleRotate(scale_limit=(-0.3, 0.3), shift_limit=(-0.3, 0.3),
                                   rotate_limit=(0, 0), border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2),
                A.RandomCrop(height=480, width=640, p=0.2),
                A.Resize(height=480, width=640)
            ],
            bbox_params=A.BboxParams(
                # format='pascal_voc', min_visibility=0.3, label_fields=['category_ids']),  # pascal_voc: [[x_min, y_min, x_max, y_max],[]]
                format='coco', min_visibility=0.3, label_fields=['category_ids']),  # coco: [[x_min, y_min, width, height],[]]
        )
        self.if_debug = True
        self.vis = init_visdom_(window_name=self.training_name)
        if self.if_debug:
            self.num_workers = 0
            self.batch_size = 1


if __name__ == "__main__":
    config = TaskConfig()
    print(config.training_name)


# Albumentations Memo

# A.Rotate(limit=(-90,-90), p=1, border_mode=cv2.BORDER_REFLECT_101)
# limit里面是旋转的角度, 正数代表逆时针旋转, 负数是正时针
# border_mode是旋转后的填充方式：cv2.BORDER_CONSTANT 会用黑色填充空出来的部分但是不会缩小
#                                cv2.BORDER_REPLICATE 会用复制边界像素的方式去填充空白处, 但是不会缩小
#                                cv2.BORDER_WRAP 会用原本的图像去填充空白处
#                                cv2.BORDER_REFLECT_101 会用原本的镜像去填充空白处
