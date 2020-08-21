import torch
from utils.init_visdom import init_visdom_
from data.coco_dataloader import DataSet, DataSetCoco
from config.config import TaskConfig
from utils.general_utils import update_print_loss_interval
vis = init_visdom_(window_name="classifier_test")


if __name__ == "__main__":
    config = TaskConfig()
    dataset = DataSetCoco(split="train", config=config)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=True)
    update_print_loss_interval(config, len(dataset))
    for i, (origin_img) in enumerate(train_loader):
        if i % config.print_loss_interval == config.print_loss_remainder:
            if config.use_visdom:
                origin_img_show = origin_img[0].numpy().copy()
                vis.image(origin_img_show.transpose(2, 0, 1)[::-1, ...], win="**********train_origin_img**********", opts={
                          'title': '**********train_origin_img {} * {}**********'.format(origin_img_show.shape[1], origin_img_show.shape[0])})
