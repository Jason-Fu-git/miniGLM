import os

import torch
import numpy as np

train_data = None
val_data = None
train_mask = None
val_mask = None


def init_data_pretrain(dataset):
    """
    初始化预训练集
    :param dataset: 数据集名称， e.g. 'shediao'
    :return: None, 使用global变量train_data, val_data
    """
    global train_data, val_data

    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


def init_data_sft(dataset):
    """
       初始化预训练集
       :param dataset: 数据集名称， e.g. 'shediao'
       :return: None, 使用global变量train_data, val_data, train_mask, val_mask, dim1
       """
    global train_data, val_data, train_mask, val_mask

    data_dir = os.path.join('data', dataset)
    dim1 = np.memmap(os.path.join(data_dir, 'dim1.bin'), dtype=np.uint16, mode='r')[0]  # 读取填充长度
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r').reshape(-1, dim1)
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r').reshape(-1, dim1)
    train_mask = np.memmap(os.path.join(data_dir, 'train_mask.bin'), dtype=np.uint16, mode='r').reshape(-1, dim1)
    val_mask = np.memmap(os.path.join(data_dir, 'val_mask.bin'), dtype=np.uint16, mode='r').reshape(-1, dim1)


def get_batch_pretrain(split, batch_size, block_size, device):
    """
    获取预训练集的batch
    :param split: 值为train或val，前者在训练时使用，后者在验证时使用
    :return: x, y, loss_mask 批次和损失函数掩码
    """
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])  # 向后错开一个字，作为标签
    loss_mask = torch.ones_like(x, dtype=torch.float64)

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device,
                                                                                          non_blocking=True), loss_mask.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask


def get_batch_sft(split, batch_size, block_size, device):
    """
    获取sft数据的批次
    :param split: 值为train或val，前者在训练时使用，后者在验证时使用
    :return: x, y, loss_mask 批次和损失函数掩码
    """
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    mask = train_mask if split == 'train' else val_mask
    ix = torch.randint(data.shape[0], (batch_size,))  # 一次训练batch_size 个问与答
    x_ls = []
    y_ls = []
    loss_ls = []
    for i in ix:
        start = torch.randint(data.shape[1] - block_size, (1,))[0]  # 随机选取一个起始位置
        x_ls.append(torch.from_numpy((data[i, start:start + block_size]).astype(np.int64)))
        y_ls.append(torch.from_numpy((data[i, start + 1:start + 1 + block_size]).astype(np.int64)))  # 向后错开一句，作为标签
        loss_ls.append(torch.from_numpy((mask[i, start + 1:start + 1 + block_size]).astype(np.int64)))  # 标签的mask
    x = torch.stack(x_ls)
    y = torch.stack(y_ls)
    loss_mask = torch.stack(loss_ls)

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device,
                                                                                          non_blocking=True), loss_mask.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask
