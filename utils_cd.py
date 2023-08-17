import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import utils

from datasets.levir_dataset import LevirDataset


def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset'):
    

    data_set = LevirDataset(data_name)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=2)

    return dataloader


def get_loaders(args):

    data_name = args.data_name
    split = args.split
    split_val = 'val'
    if args.dataset == 'LEVIR':
        training_set = LevirDataset(data_name)
        val_set = LevirDataset(data_name, split='val')
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])