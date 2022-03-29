##############################################################################
# MAF implementation for Few-shot Domain Adaptation
##############################################################################
import json
import os
import shutil
import time
import numpy as np

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from data.prepare_data import generate_dataloader  # Prepare the data and dataloader
from models.resnet import resnet  # The model construction
from models.vgg import vgg
from opts import opts  # The options for the project
from trainer import train  # For the training process
from trainer import validate  # For the validate (test) process
from trainer import adjust_learning_rate
from trainer import AverageMeter # save results of each epoch
from models.DomainClassifierTarget import DClassifierForTarget
from models.DomainClassifierSource import DClassifierForSource
from models.EntropyMinimizationPrinciple import EMLossForTarget
from models.ISDALoss import ISDALoss
from models.IPMALoss import IPMALoss
import ipdb
import math

best_prec1 = 0

def main(args):
    global best_prec1
    current_epoch = 0
    epoch_count_dataset = 'source'

    print("data_path_source: ", args.data_path_source, args.arch)
    if args.arch == 'alexnet':
        raise ValueError('the request arch is not prepared', args.arch)
    elif args.arch.find('resnet') != -1:
        print("Building ResNet model ...")
        model = resnet(args)
    elif args.arch.find('vgg') != 1:
        print("Building VGG model ...")
        model = vgg(args)
        print("VGG backbone!")
    else:
        raise ValueError('Unavailable model architecture!!!')
    # define-multi GPU
    model = torch.nn.DataParallel(model).cuda()
    criterion_classifier_target = DClassifierForTarget(nClass=args.num_classes).cuda() # dual-classifier, softmax for target part
    criterion_classifier_source = DClassifierForSource(nClass=args.num_classes).cuda() # dual-classifier, softmax for source part
    criterion_em_target = EMLossForTarget(nClass=args.num_classes).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    if args.arch.find('resnet') != -1:
        feat_dim = 2048
    elif args.arch.find('vgg') != -1:
        feat_dim = 4096
    else:
        raise Exception("No arch defined!")
    isda_criterion = ISDALoss(feat_dim, args.num_classes).cuda() # upperbound of augmented data for target classifier
    isda_criterion_Cs = ISDALoss(feat_dim, args.num_classes).cuda() # upperbound of augmented data for source classifier

    results = {'batch_time': AverageMeter(), 'data_time':AverageMeter(), 'losses_classifier':AverageMeter(), \
               'losses_G':AverageMeter(), 'top1_source': AverageMeter(), 'top1_target':AverageMeter(),
               'loss_cC': AverageMeter(), 'loss_dC': AverageMeter(), 'loss_aug': AverageMeter(), 'loss_cG': AverageMeter(),
               'loss_dG': AverageMeter(), 'loss_cC': AverageMeter(), }
    record = {'loss_C': [], 'loss_G': [], 'loss_c^C': [], 'loss_d^C': [], 'loss_aug': [], 'loss_c^G': [], 'loss_d^G': [],
              'Cs_test_acc': [], 'Ct_test_acc': [], 'CST_test_acc': [],
              'diff_prob_norm': [], 'diff_pred_same': []
              }

    # To apply different learning rate to different layer
    if args.arch == 'alexnet':
        optimizer = torch.optim.SGD([
            # {'params': model.module.features1.parameters(), 'name': 'pre-trained'},
            {'params': model.module.features2.parameters(), 'name': 'pre-trained'},
            {'params': model.module.classifier.parameters(), 'name': 'pre-trained'},
            {'params': model.module.fc.parameters(), 'name': 'new-added'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.arch.find('resnet') != -1:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([
                {'params': model.module.conv1.parameters(), 'name': 'pre-trained'},
                {'params': model.module.bn1.parameters(), 'name': 'pre-trained'},
                {'params': model.module.layer1.parameters(), 'name': 'pre-trained'},
                {'params': model.module.layer2.parameters(), 'name': 'pre-trained'},
                {'params': model.module.layer3.parameters(), 'name': 'pre-trained'},
                {'params': model.module.layer4.parameters(), 'name': 'pre-trained'},
                #{'params': model.module.fc.parameters(), 'name': 'pre-trained'}
                {'params': model.module.fc.parameters(), 'name': 'new-added'}
            ],
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam([
                {'params': model.module.conv1.parameters(), 'name': 'pre-trained'},
                {'params': model.module.bn1.parameters(), 'name': 'pre-trained'},
                {'params': model.module.layer1.parameters(), 'name': 'pre-trained'},
                {'params': model.module.layer2.parameters(), 'name': 'pre-trained'},
                {'params': model.module.layer3.parameters(), 'name': 'pre-trained'},
                {'params': model.module.layer4.parameters(), 'name': 'pre-trained'},
                # {'params': model.module.fc.parameters(), 'name': 'pre-trained'}
                {'params': model.module.fc.parameters(), 'name': 'new-added'}
            ],
                lr=args.lr,
                weight_decay=args.weight_decay)
        else:
            raise Exception("Optimizer is not defined!")
    elif args.arch.find('vgg') != -1:
        # print(model.module)
        optimizer = torch.optim.Adam([
            {'params': model.module.vgg_conv.parameters(), 'name': 'pre-trained'},
            {'params': model.module.vgg_feat.parameters(), 'name': 'pre-trained'},
            # {'params': model.module.fc.parameters(), 'name': 'pre-trained'}
            {'params': model.module.fc.parameters(), 'name': 'new-added'}
        ],
            lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        raise ValueError('Unavailable model architecture!!!')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from is not exited', args.resume)
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    source_train_loader, source_val_loader, target_train_loader, val_loader = generate_dataloader(args)
    print("Source train: ", len(source_train_loader.dataset), " target train: ", len(target_train_loader.dataset), \
          " source val: ", len(source_val_loader.dataset), " target validation: ", len(val_loader.dataset))



    #test only
    if args.test_only:
        validate(val_loader, model, criterion, -1, args, record)
        return
    # start time
    cs_weight_norm = torch.mean(torch.norm(model.module.fc.weight[:args.num_classes, :], p=2, dim=1)).item()
    ct_weight_norm = torch.mean(torch.norm(model.module.fc.weight[args.num_classes:, :], p=2, dim=1)).item()
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.write("Initialization Cs weight norm: %4f , Ct weight norm: %4f  " % (cs_weight_norm,ct_weight_norm))
    log.write('\n-------------------------------------------\n')
    log.close()

    source_train_loader_batch = enumerate(source_train_loader)
    target_train_loader_batch = enumerate(target_train_loader)
    batch_number_s = len(source_train_loader)
    batch_number_t = len(target_train_loader)
    if batch_number_s < batch_number_t:
        epoch_count_dataset = 'target' # in order to loop all larger one's data batches
    else:
        epoch_count_dataset = 'source'
    a = 0


    ''' Save info during training and test:
        1. losses
        2. Cs and Ct performance
    '''

    prec1 = validate(val_loader, model, criterion, current_epoch, args, record)
    while (current_epoch < args.epochs):
        # for each epoch
        adjust_learning_rate(optimizer, current_epoch, args)
        source_train_loader_batch, target_train_loader_batch, current_epoch, new_epoch_flag \
            = train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch,
                    model, criterion_classifier_source, criterion_classifier_target, criterion_em_target, criterion,
                    isda_criterion, ipma_criterion, isda_criterion_Cs, ipma_criterion_Cs, optimizer, current_epoch,
                    epoch_count_dataset, results, args, record)
        if a <5:
            print("Succeeded 1 iter...")
            a += 1
        # evaluate on the val data
        if new_epoch_flag:
            record['loss_C'].append(results['losses_classifier'].avg)
            record['loss_G'].append(results['losses_G'].avg)
            record['loss_c^C'].append(results['loss_cC'].avg)
            record['loss_d^C'].append(results['loss_dC'].avg)
            record['loss_c^G'].append(results['loss_cG'].avg)
            record['loss_d^G'].append(results['loss_dG'].avg)
            record['loss_aug'].append(results['loss_aug'].avg)

            # Create new ISDA / IPMA object, refresh Ave, Covariance, tAve
            # Do not renew from zeros at every epoch
            isda_criterion = ISDALoss(feat_dim, args.num_classes).cuda()
            ipma_criterion = IPMALoss(feat_dim, args.num_classes).cuda()
            isda_criterion_Cs = ISDALoss(feat_dim, args.num_classes).cuda()
            ipma_criterion_Cs = IPMALoss(feat_dim, args.num_classes).cuda()
            results = {'batch_time': AverageMeter(), 'data_time': AverageMeter(), 'losses_classifier': AverageMeter(), \
                       'losses_G': AverageMeter(), 'top1_source': AverageMeter(), 'top1_target': AverageMeter(),
                       'loss_cC': AverageMeter(), 'loss_dC': AverageMeter(), 'loss_aug': AverageMeter(),
                       'loss_cG': AverageMeter(), 'loss_dG': AverageMeter(), 'loss_cC': AverageMeter(), }

            # Print results and validate
            print('Current epoch: ', current_epoch, ' lam: ',
                  1 * (2 / (1 + math.exp(-1 * 10 * current_epoch / args.epochs)) - 1), ' lr: ', optimizer.param_groups[-1]['lr'])
            print("Cs weights: ",torch.mean(torch.norm(model.module.fc.weight[:args.num_classes, :], p=2, dim=1)).item(), \
                  "Ct weights: ",torch.mean(torch.norm(model.module.fc.weight[args.num_classes:, :], p=2, dim=1)).item())

            if (current_epoch + 1) % args.test_freq == 0 or current_epoch == 0:
                print('Validation...')
                prec1 = validate(val_loader, model, criterion, current_epoch, args, record)
                # record the best prec1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                if is_best:
                    log = open(os.path.join(args.log, 'log.txt'), 'a')
                    log.write('     Best acc: %3f' % (best_prec1))
                    log.close()

                    save_checkpoint({
                        'epoch': current_epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, args)
            np.save(os.path.join(args.log, 'ncf_'+str(args.nspc)+'_'+args.src+'2'+args.tar+'record.npy'), record)

    # end time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()


def save_checkpoint(state, is_best, args):
    # filename = str(state['epoch'])+'_checkpoint.pth.tar'
    filename = 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)

    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    global args
    args = opts()

    dataset = "Office"
    class_num = 31
    way =class_num
    datapath = "./Dataset/Office/images/"
    src = "amazon"
    tar = "dslr"

    # dataset = "OfficeHome"
    # class_num = 65
    # way =class_num
    # datapath = "./Dataset/OfficeHome/images/"
    # src = "Art"
    # tar = "Product"

    split = 1000
    shot = 3

    log_folder = "shot{}_{}_split{}".format(shot, dataset, split)

    args.data_path_source = datapath
    args.data_path_target_tr = datapath
    args.data_path_target = datapath
    args.src = src
    args.tar_tr = tar
    args.tar = tar
    args.epochs = 100
    args.num_classes = class_num

    args.gamma = 0.1
    args.weight_decay = 1e-4
    args.workers = 4
    args.pretrained = True
    args.flag = 'symnet'
    args.log = log_folder
    args.batch_size_s = 16
    args.batch_size_t = 16
    args.way = way
    args.nspc = 20
    args.shot = shot
    args.dataset = dataset
    args.split = split
    args.arch = 'resnet50'
    args.optimizer = 'adam'
    args.lr = 0.0001
    args.print_freq = 1
    args.test_freq = 1

    main(args)
