import os
import torch
from tqdm import tqdm
import torch.optim as optim
from config.config import TaskConfig
from utils.init_visdom import init_visdom_
from network.efficientnet_pytorch import EfficientNet
from data.folder_dataloader import DataSet
from utils.general_utils import seed_torch, save_log
vis = init_visdom_(window_name="centernet_new")

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


class Evaltor(object):
    def __init__(self):
        # ---------------------------------  Set Val Set  ---------------------------------#
        self.val_data = DataSet(config.val_data_root, if_training=False)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=config.batch_size,
                                                      shuffle=True, pin_memory=True, num_workers=config.num_workers)


class Trainer(object):
    def __init__(self):
        # ------------------------------   Set Random Sed   -------------------------------#
        if config.set_seed:
            seed_torch()  # Fix random seed for try different Hyper-Parameter
            save_log(config.log_file_path, "Set Random Sed")
        # -------------------------------  Set Training Set  ------------------------------#
        # train
        self.train_data = DataSet(
            config.train_data_root, transform=config.transform, if_training=True)
        config.class_num = self.train_data.class_num
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=config.batch_size,
                                                        shuffle=True, pin_memory=True, num_workers=config.num_workers)
        # --------------------------------  Set Evaltor   ----------------------------------#
        self.evaltor = Evaltor()

        # -------------------------------   Init Network  ----------------------------------#
        if "efficientnet" in config.model_name:
            self.model = EfficientNet.from_pretrained(config.model_name, num_classes=config.class_num).cuda(
                config.gpu_num[0])

        # -------------------------------   Set Optimizer ----------------------------------#
        # Use SGD
        if config.which_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=config.base_lr, momentum=0.9, weight_decay=1e-5)
        # Use ADAM
        elif config.which_optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=config.base_lr)

    def train(self):
        self.model.train()
        for i, (aug_img_tensor, labels, aug_img, origin_img) in enumerate(self.train_loader):
            if i % config.print_loss_interval == config.print_loss_remainder:
                if config.use_visdom:
                    origin_img_show = origin_img[0].numpy().copy()
                    aug_img_show = aug_img[0].numpy().copy()
                    vis.image(origin_img_show.transpose(2, 0, 1)[::-1, ...], win="**********train_origin_img**********", opts={
                        'title': '**********train_origin_img {} * {}**********'.format(origin_img_show.shape[1], origin_img_show.shape[0])})

                    vis.image(aug_img_show.transpose(2, 0, 1)[::-1, ...], win="**********train_aug_img**********", opts={
                        'title': '**********train_aug_img {} * {}**********'.format(aug_img_show.shape[1], aug_img_show.shape[0])})

            # bar = tqdm(iter(self.train_loader), ascii=True)
            # for imgs, labels in bar:
            #     imgs = imgs.cuda()
            #     labels = labels.cuda()
            #     print(labels)


if __name__ == "__main__":
    config = TaskConfig()
    tainer = Trainer()
    tainer.train()
