import os
import torch
from tqdm import tqdm
import torch.optim as optim
from config.config import TaskConfig
from data.coco_dataloader import DataSetCoco
from utils.general_utils import seed_torch, save_log, update_print_loss_interval
import networks.centernet.pose_dla_dcn as dladcn
from utils.img_utils import visual_training_process
from networks.losses import CtdetLoss

# class Evaltor(object):
#     def __init__(self):
#         # ---------------------------------  Set Val Set  ---------------------------------#
#         self.val_data = DataSet(config.val_data_root, if_training=False)
#         self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=config.batch_size,
#                                                       shuffle=True, pin_memory=True, num_workers=config.num_workers)


class Trainer(object):
    def __init__(self):
        # -------------------------------   Init Network  ---------------------------------#
        self.config = TaskConfig()

        # ------------------------------   Set Random Sed   -------------------------------#
        if self.config.set_seed:
            seed_torch()  # Fix random seed for try different Hyper-Parameter
            save_log(self.config.log_file_path, "Set Random Sed")
        # -------------------------------  Set Training Set  ------------------------------#
        self.dataset = DataSetCoco(split="train", config=self.config)
        update_print_loss_interval(self.config, len(self.dataset))
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, pin_memory=True)
        # -------------------------------  Set Evaltor   ----------------------------------#
        # self.evaltor = Evaltor()

        # ------------------------------   Init Network  ----------------------------------#
        if "dla" in self.config.model_name:
            self.network = dladcn.getDlaDCN(34, heads={
                "hm": self.config.class_num, "reg": 2, "wh": 2}).cuda(self.config.gpu_num[0])

        # ------------------------------   Set Optimizer ----------------------------------#
        # Use SGD
        if self.config.which_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.network.parameters(), lr=self.config.base_lr, momentum=0.9, weight_decay=1e-5)
        # Use ADAM
        elif self.config.which_optimizer == "adam":
            self.optimizer = optim.Adam(
                self.network.parameters(), lr=self.config.base_lr)
        # -------------------------------   Init Loss   -----------------------------------#
        self.network_loss = CtdetLoss(self.config).cuda()
        self.train_epoch_batchs = len(self.train_loader)

    def train(self):
        self.network.train()
        for epoch in range(self.config.train_epoch):
            for indbatch, batch in enumerate(self.train_loader):
                # visualize bounding box and/or heatmap
                if (indbatch % self.config.print_loss_interval) == self.config.print_loss_remainder:
                    visual_training_process(self.config, batch, self.dataset.coco)
                    print(
                        f"lr: {self.optimizer.param_groups[0]['lr']:.6f}, epoch: {epoch_flt:.2f} loss: {loss_stats['loss'].item():.4f}, hm_loss: {loss_stats['hm_loss'].item():.4f}, "
                        f"xy_loss: {loss_stats['off_loss'].item():.4f}, wh_loss: {loss_stats['wh_loss'].item():.4f}"
                    )
                cuda_jump_list = ['auged_bbox',
                                  "auged_img", "category_name_list"]
                for k in batch:
                    if k not in cuda_jump_list:
                        batch[k] = batch[k].cuda(self.config.gpu_num[0])
                outputs = self.network(batch['input'])
                loss, loss_stats = self.network_loss(outputs, batch)
                loss = loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_flt = epoch + indbatch / float(self.train_epoch_batchs)


if __name__ == "__main__":
    tainer = Trainer()
    tainer.train()
