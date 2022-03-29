import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_test_split(shot, labels, class_num=None):
    uni_label, indexes, counts = np.unique(labels, return_index=True, return_counts=True)
    # labels should be already sorted
    n = len(labels)
    if not class_num:
        class_num = len(uni_label)

    split = np.zeros(class_num * shot) - 1 # initialize as -1s
    for c in range(class_num):
        c_start_idx = indexes[c]
        c_count = counts[c]
        c_end_idx = c_start_idx + c_count
        c_subinds = np.arange(c_start_idx, c_end_idx)
        np.random.shuffle(c_subinds)

        cn = len(c_subinds)
        if cn >= shot:
            c_rand_samples = c_subinds[:shot]
            split_start_idx = c * shot
            split_end_idx = (c + 1) * shot
            split[split_start_idx:split_end_idx] = c_rand_samples
        else:  # cn < shot
            print("class ", c, " counts < shot... make up ", shot - cn, " duplicate samples")
            c_orig_samples = c_subinds[:cn]
            c_mkup_samples = np.random.randint(c_start_idx, c_end_idx, shot - cn) # randomly repeatedly sampling
            c_rand_samples = np.concatenate((c_orig_samples, c_mkup_samples))
            split_start_idx = c * shot
            split_end_idx = (c + 1) * shot
            split[split_start_idx:split_end_idx] = c_rand_samples

    if np.min(split) < 0:
        raise Exception("Error with -1 !!!")
    train_inds = split.astype(int)
    test_inds = np.array([i for i in range(n) if i not in train_inds]).astype(int)
    return train_inds, test_inds

def generate_dataloader(args):
    # Data loading code
    traindir_source = os.path.join(args.data_path_source, args.src) # source folder
    traindir_target = os.path.join(args.data_path_target_tr, args.tar_tr) # target folder
    valdir = os.path.join(args.data_path_target, args.tar) # target folder
    print(traindir_source, traindir_target, valdir)
    if not (os.path.isdir(traindir_source) and os.path.isdir(traindir_target) and os.path.isdir(valdir)):
        # split_train_test_images(args.data_path)
        raise ValueError('Null path of data!!!')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    source_train_dataset = datasets.ImageFolder(
        traindir_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source_val_dataset = datasets.ImageFolder(
        traindir_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    print("Source total samples: ", len(source_train_dataset))

    ''' Source few'''
    dataset = args.dataset
    src_name = args.src  # 'amazon'
    way = args.way  # 31

    if dataset == 'Office':
        # for Office, change source training / val data numbers
        if src_name == 'amazon': # Only for Office, source select
            # ncf = 20, follow prior FSDA works
            ncf = args.nspc
            print("Dealing with amazon source samples per class: ", ncf)
        else:
            ncf = 8
        src_few_inds, src_val_inds = train_test_split(shot=ncf, labels=source_train_dataset.targets, class_num=way)
        src_few_dataset = torch.utils.data.Subset(source_train_dataset, src_few_inds)
        # For Office, rest source data can be used as validation
        src_val_dataset = torch.utils.data.Subset(source_val_dataset, src_val_inds)

        source_train_loader = torch.utils.data.DataLoader(
            src_few_dataset, batch_size=args.batch_size_s, shuffle=True,
            drop_last=True, num_workers=args.workers, pin_memory=True, sampler=None
        )

    else: # Otherwise, for OfficeHome, not need to sample data
        source_train_loader = torch.utils.data.DataLoader(
            source_train_dataset, batch_size=args.batch_size_s, shuffle=True,
            drop_last=True, num_workers=args.workers, pin_memory=True, sampler=None
        )
        # For OfficeHome, no source validation data available
        src_val_dataset = source_val_dataset


    source_val_loader = torch.utils.data.DataLoader(
        src_val_dataset, batch_size=args.batch_size_s, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )
    # -----------------------------------------------------------------------

    '''  Target dataset     '''
    target_train_dataset = datasets.ImageFolder(
        traindir_target,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    target_val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    #print("Target total samples: ", len(target_train_dataset))

    ''' Few set target training split '''
    dataset = args.dataset
    tar_name = args.tar#'dslr'
    way = args.way#31
    shot = args.shot#3

    tar_few_inds, tar_val_inds = train_test_split(shot=shot, labels=target_train_dataset.targets, class_num=way)
    tar_few_dataset = torch.utils.data.Subset(target_train_dataset, tar_few_inds)
    tar_val_dataset = torch.utils.data.Subset(target_val_dataset, tar_val_inds)


    target_train_loader = torch.utils.data.DataLoader(
        tar_few_dataset, batch_size=args.batch_size_t, shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True, sampler=None
    )
    ''' target val dataset '''
    # -----------------------------------------------------------------------
    target_val_loader = torch.utils.data.DataLoader(
        tar_val_dataset, batch_size=args.batch_size_t, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    return source_train_loader, source_val_loader, target_train_loader, target_val_loader

