import datetime
import os
import torch
import shutil
import numpy as np

# ---------------------------   Time  ------------------------------#
def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# ---------------------------   Log   ------------------------------#
# Write Log into txt
def save_log(txt_path, string_contant):
    with open(txt_path, "a") as f:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S:')
        string_contant = nowTime + string_contant + "\n"
        print(string_contant)
        f.writelines(string_contant)


# ---------------------------   File   -----------------------------#
def check_file_without_delete(path):
    if os.path.exists(path):
        print("Already Exist: {}".format(path))
    else:
        print("Do not exist：", path)
        os.system("touch {}".format(path))
        print("Create： ", path)


def check_file_with_delete(path):
    if os.path.exists(path):
        print("Already Exist: {}".format(path))
        os.system("rm {} -r".format(path))
        print("Delete： ", path)
        os.system("touch {}".format(path))
        print("Create： ", path)
    else:
        print("Do not exist：", path)
        os.system("touch {}".format(path))
        print("Create： ", path)


def check_path_with_delete(path):
    if os.path.exists(path):
        print("Already Exist: {}".format(path))
        shutil.rmtree(path)
        print("Delete： ", path)
        os.makedirs(path)
        print("Create： ", path)
    else:
        print("Do not exist：", path)
        os.makedirs(path)
        print("Create： ", path)


def check_path_without_delete(path):
    if os.path.exists(path):
        print("Already Exist: {}".format(path))
    else:
        os.makedirs(path)
        print("Create： ", path)


# --------------------------   Random   ----------------------------#
def seed_torch(seed=3):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# --------------------------   Print    -----------------------------#
# 跟新打印loss的间隔
def update_print_loss_interval(config, length_of_dataset):
    if (length_of_dataset / config.batch_size) < 2 * 10:
        config.print_loss_interval = 1
        config.print_loss_remainder = 0
    elif 2 * 10 <= (length_of_dataset / config.batch_size) < 2 * 100:
        config.print_loss_interval = 10
        config.print_loss_remainder = 9
    elif 2 * 100 <= (length_of_dataset / config.batch_size) < 2 * 1000:
        config.print_loss_interval = 100
        config.print_loss_remainder = 99
    elif 2 * 1000 <= (length_of_dataset / config.batch_size) < 2 * 10000:
        config.print_loss_interval = 1000
        config.print_loss_remainder = 999
    elif (length_of_dataset / config.batch_size) >= 2 * 10000:
        config.print_loss_interval = 10000
        config.print_loss_remainder = 9999


if __name__ == "__main__":
    log_path = "./1.txt"
    train_info = "loss: {}".format(round(0.2222123123124, 6))
    save_log(log_path, train_info)
