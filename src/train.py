import os
import torch
from tqdm import tqdm
import torch.optim as optim
from config.config import TaskConfig
from data.coco_dataloader import DataSetCoco
from utils.general_utils import seed_torch, save_log, update_print_loss_interval


class Evaltor(object):
    def __init__(self):
        # ---------------------------------  Set Val Set  ---------------------------------#
        self.val_data = DataSet(config.val_data_root, if_training=False)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=config.batch_size,
                                                      shuffle=True, pin_memory=True, num_workers=config.num_workers)


class Trainer(object):
    def __init__(self, config):
        # ------------------------------   Set Random Sed   -------------------------------#
        if config.set_seed:
            seed_torch()  # Fix random seed for try different Hyper-Parameter
            save_log(config.log_file_path, "Set Random Sed")
        # -------------------------------  Set Training Set  ------------------------------#
        dataset = DataSetCoco(split="train", config=config)
        update_print_loss_interval(config, len(dataset))
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
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
        for indbatch, batch in enumerate(self.train_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda(config.gpu_num[0])

if __name__ == "__main__":
    config = TaskConfig()
    tainer = Trainer(config)
    tainer.train()
