import time
import torch
import os
import math
import ipdb
import torch.nn.functional as F
from data.augment_data import augment_data
# from trainer import AverageMeter

def train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch, model,
          criterion_classifier_source, criterion_classifier_target, criterion_em_target, criterion, isda_criterion,
          ipma_criterion, isda_criterion_Cs, ipma_criterion_Cs, optimizer, epoch, epoch_count_dataset, results, args,
          record):

    batch_time = results['batch_time']
    data_time = results['data_time']
    losses_classifier = results['losses_classifier']
    losses_G = results['losses_G']
    top1_source = results['top1_source']
    top1_target = results['top1_target']

    loss_cC = results['loss_cC']
    loss_dC = results['loss_dC']
    loss_aug = results['loss_aug']
    loss_cG = results['loss_cG']
    loss_dG = results['loss_dG']




    model.train()

    new_epoch_flag = False
    end = time.time()
    # print("prepare data...")
    try:
        s_batch_id, s_data = source_train_loader_batch.__next__()
        (input_source, target_source) = s_data
        if s_batch_id >= len(source_train_loader):
            # print("Out of source batch indexes...")
            raise StopIteration
    except StopIteration:
        # print("Ran out of source batches...")
        if epoch_count_dataset == 'source': # running out of source loader, start a new epoch
            epoch = epoch + 1
            new_epoch_flag = True
        source_train_loader_batch = enumerate(source_train_loader)
        (input_source, target_source) = source_train_loader_batch.__next__()[1]
    try:
        # print("start target...")
        t_batch_id, t_data = target_train_loader_batch.__next__()
        (input_target, target_target) = t_data
        if t_batch_id >= len(target_train_loader):
            # print("Out of target batch indexes...")
            raise StopIteration
    except StopIteration:
        # print("Ran out of target batches...")
        if epoch_count_dataset == 'target':
            # print("epoch count target")
            epoch = epoch + 1
            new_epoch_flag = True
        target_train_loader_batch = enumerate(target_train_loader)
        (input_target, target_target) = target_train_loader_batch.__next__()[1]
    data_time.update(time.time() - end)
    # print('Got data loader! Calculate losses...')
    #### Source data
    target_source_temp = target_source + args.num_classes  # Ys + C
    target_source_temp = target_source_temp.cuda(non_blocking=True)
    target_source_temp_var = torch.autograd.Variable(target_source_temp) #### labels for target classifier, +num_classes

    target_source = target_source.cuda(non_blocking=True) # Ys
    input_source_var = torch.autograd.Variable(input_source) # Xs
    target_source_var = torch.autograd.Variable(target_source) ######## labels for source classifier.

    ### Target data
    target_target_temp = target_target+ args.num_classes # Yt + C
    target_target_temp = target_target_temp.cuda(non_blocking=True)
    target_target_temp_var = torch.autograd.Variable(target_target_temp)  #### labels for source classifier, +num_classes

    target_target = target_target.cuda(non_blocking=True) # Yt
    input_target_var = torch.autograd.Variable(input_target) # Xt
    target_target_var = torch.autograd.Variable(target_target)  ######## labels for target classifier.

    ############################################ for source samples
    output_source, s_feats = model(input_source_var)
    output_target, t_feats = model(input_target_var)
    # s_feats = F.normalize(s_feats, dim=1, p=2)
    # t_feats = F.normalize(t_feats, dim=1, p=2)

    # (1) Supervised learning ---> L_c^C
    # (1.1) source
    loss_task_s_Cs = criterion(output_source[:,:args.num_classes], target_source_var) # cross-entropy, for source data, for Cs,Ct separately
    loss_task_s_Ct = criterion(output_source[:,args.num_classes:], target_source_var)
    # (1.2) target
    loss_task_t_Cs = criterion(output_target[:, :args.num_classes], target_target_var) # cross-entropy, for target data, for Cs,Ct separately
    loss_task_t_Ct = criterion(output_target[:, args.num_classes:], target_target_var)

    #(2) Domain Confusion loss ---> L_d^C, domain classifier calculated on C_st, only used to recognize source/target, sum to domain discriminator
    loss_domain_st_Cst_part1 = criterion_classifier_source(output_source)
    loss_domain_st_Cst_part2 = criterion_classifier_target(output_target)

    # (3) Generator discrimination confusion ---> L_d^G: make Cs and Ct predict similar for both source/target, sum to domain discriminator
    loss_domain_t_G = 0.5 * criterion_classifier_target(output_target) \
                      + 0.5 * criterion_classifier_source(output_target)
    loss_domain_s_G = 0.5 * criterion_classifier_target(output_source) \
                      + 0.5 * criterion_classifier_source(output_source)
    # (4) Generator category confusion ---> L_c^G : confuse Cs/Ct, but not domain discminator, classifier CELoss for both Cs and Ct
    loss_category_s_G = 0.5 * criterion(output_source, target_source_var) \
                        + 0.5 * criterion(output_source, target_source_temp_var)
    loss_category_t_G = 0.5 * criterion(output_target, target_target_var) \
                        + 0.5 * criterion(output_target, target_target_temp_var)

    # (5) Implicit Augmentation
    # (5.1) Marginalized Augmentation
    alpha0 = 0.5  # 0.5
    alpha = alpha0 * (epoch / args.epochs)
    beta0 = 1.0 # 0.5
    beta = beta0 * (epoch / args.epochs)
    loss_ISDA_Ct = isda_criterion(model, input_source_var, input_target_var, s_feats, t_feats,
                                             target_source_var, target_target_var, \
                                             output_target[:, args.num_classes:], cls='Ct', ratio_alpha=alpha, ratio_beta=beta)
    loss_ISDA_Cs = isda_criterion_Cs(model, input_source_var, input_source_var, s_feats, s_feats,
                                  target_source_var, target_source_var, \
                                  output_target[:, :args.num_classes], cls='Cs', ratio_alpha=alpha, ratio_beta=beta)

    lam = 0.1*(2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1) # gamma


    if args.flag == 'no_em':
        raise Exception("no_em is not defined!")
    elif args.flag == 'symnet':    #
        loss_aug_C = loss_ISDA_Ct + loss_ISDA_Cs 
        loss_sup_C = 1*(loss_task_s_Cs + loss_task_s_Ct)  + loss_task_t_Cs+  loss_task_t_Ct # ---> L_c^C
        loss_domain_C = loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2 # ---> L_d^C
        loss_sup_G = loss_category_s_G + loss_category_t_G #  ---> L_c^G
        loss_domain_G = loss_domain_s_G + loss_domain_t_G #  ---> L_d^G

        if epoch < -1:
            loss_classifier = loss_sup_C
            loss_G = loss_sup_C
        else:
            loss_classifier = loss_sup_C + loss_domain_C + lam * loss_aug_C 
            loss_G = loss_sup_G + 1 * loss_domain_G #+ 0.1 * loss_st_em
    else:
        raise ValueError('unrecognized flag:', args.flag)

    # mesure accuracy and record loss

    prec1_source, _ = accuracy(output_source.data[:, :args.num_classes], target_source, topk=(1, 5))
    prec1_target, _ = accuracy(output_target.data[:, args.num_classes:], target_target, topk=(1, 5))

    input_size = input_source.size(0) + input_target.size(0) 
    losses_classifier.update(loss_classifier.item(), input_size)
    losses_G.update(loss_G.item(), input_size)
    top1_source.update(prec1_source[0], input_source.size(0))
    top1_target.update(prec1_target[0], input_source.size(0))

    loss_cC.update(loss_sup_C.item(), input_source.size(0))
    loss_dC.update(loss_domain_C.item(), input_source.size(0))
    loss_aug.update(loss_aug_C.item(), input_source.size(0))
    loss_cG.update(loss_sup_G.item(), input_source.size(0))
    loss_dG.update(loss_domain_G.item(), input_source.size(0))

    #compute gradient and do SGD step
    # print("Optimization...")
    optimizer.zero_grad()
    loss_classifier.backward(retain_graph=True)
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_classifier = temp_grad
    
    optimizer.zero_grad()
    loss_G.backward()
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_featureExtractor = temp_grad
    
    count = 0
    for param in model.parameters():
        temp_grad = param.grad.data.clone()
        temp_grad.zero_()

        if args.arch == 'resnet101':
            feats_layer = 312 # resnet101: 312
        elif args.arch == 'resnet50':
            feats_layer = 159 #resnet50:159 the feautre extractor of the ResNet-50
        elif args.arch == 'vgg16':
            feats_layer = 30
        else:
            raise Exception("Can not recognize backbone feature extractor!")

        if count < feats_layer:
            temp_grad = temp_grad + grad_for_featureExtractor[count]
        else:
            temp_grad = temp_grad + grad_for_classifier[count]
        temp_grad = temp_grad
        param.grad.data = temp_grad
        count = count + 1
    optimizer.step()
    batch_time.update(time.time() - end)
    end = time.time()

    if (epoch + 1) % args.print_freq == 0 or epoch == 0:
        # print(epoch, args.epochs, losses_classifier.avg, losses_classifier.val, top1_source.avg, top1_target.val)
        print('Train: [{0}/{1}]\t'
              'Loss@C {loss_c.val:.4f} ({loss_c.avg:.4f})\t'
              'Loss@G {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
              'top1tr_s_CS {top1S.val:.3f} ({top1S.avg:.3f})\t'
              'top1tr_t_CT {top1T.val:.3f} ({top1T.avg:.3f})'.format(
            epoch, args.epochs, loss_c=losses_classifier, loss_g=losses_G, top1S=top1_source, top1T=top1_target))
        if new_epoch_flag:
            cs_weight_norm = torch.mean(torch.norm(model.module.fc.weight[:args.num_classes, :], p=2, dim=1)).item()
            ct_weight_norm = torch.mean(torch.norm(model.module.fc.weight[args.num_classes:, :], p=2, dim=1)).item()

            print('Train: [{0}/{1}]\t'
                  'Loss@C {loss_c.val:.4f} ({loss_c.avg:.4f})\t'
                  'Loss@G {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                  'top1S {top1S.val:.3f} ({top1S.avg:.3f})\t'
                  'top1T {top1T.val:.3f} ({top1T.avg:.3f})'.format(
                epoch, args.epochs, loss_c=losses_classifier, loss_g=losses_G, top1S=top1_source, top1T=top1_target))
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            log.write("\n")
            log.write("Train:epoch: %d, loss@min: %4f, loss@max: %4f, Top1S acc: %3f, Top1T acc: %3f, Cs weight norm: %4f , Ct weight norm: %4f  " % (epoch, losses_classifier.avg, losses_G.avg, top1_source.avg, top1_target.avg,cs_weight_norm,ct_weight_norm))
            log.close()
    
    return source_train_loader_batch, target_train_loader_batch, epoch, new_epoch_flag


def validate(val_loader, model, criterion, epoch, args, record):
    batch_time = AverageMeter()
    losses_source = AverageMeter()
    losses_target = AverageMeter()
    top1_source = AverageMeter()
    top1_target = AverageMeter()
    # switch to evaluate mode
    top1_st = AverageMeter()

    prob_diff = AverageMeter()
    pred_diff = AverageMeter()
    model.eval()

    end = time.time()
    # for i, (input, target,_) in enumerate(val_loader):
    for i, (input, target) in enumerate(val_loader):
        # # ----------------------------
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input) #, volatile=True)
        target_var = torch.autograd.Variable(target) #, volatile=True)
        # compute output
        with torch.no_grad():
            output,_ = model(input_var)
        loss_source = criterion(output[:, :args.num_classes], target_var)
        loss_target = criterion(output[:, args.num_classes:], target_var)
        # measure accuracy and record loss
        prec1_source, _ = accuracy(output.data[:, :args.num_classes], target, topk=(1, 5))
        prec1_target, _ = accuracy(output.data[:, args.num_classes:], target, topk=(1, 5))
        prec1_st, _ = accuracy((output.data[:, :args.num_classes] + output.data[:, args.num_classes:]) / 2, target, topk=(1,5))

        # losses_source.update(loss_source.data[0], input.size(0))
        # losses_target.update(loss_target.data[0], input.size(0))
        losses_source.update(loss_source.data, input.size(0))
        losses_target.update(loss_target.data, input.size(0))

        top1_source.update(prec1_source[0], input.size(0))
        top1_target.update(prec1_target[0], input.size(0))
        top1_st.update(prec1_st[0], input.size(0))
        prob_diff.update(torch.mean(torch.norm(
            torch.softmax(output.data[:, :args.num_classes], dim=-1)
            -torch.softmax(output.data[:, args.num_classes:], dim=-1)
        )).item(), input.size(0))
        pred_diff.update(torch.sum(
            torch.argmax(output.data[:, :args.num_classes], dim=-1)
            == torch.argmax(output.data[:, args.num_classes:], dim=-1)
        ).item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    print(' Target Validation: * Top1@CS {top1S.avg:.3f} Top1@CT {top1T.avg:.3f}  Top1@CST {top1ST.avg:.3f}!!!'
          .format(top1S=top1_source, top1T=top1_target, top1ST=top1_st))


    # record = {'loss_total': [], 'loss_c^C': [], 'loss_c^C': [], 'loss_aug': [], 'loss_c^G': [], 'loss_c^G': [],
    #                'Cs_test_acc': [], 'Ct_test_acc': [], 'CST_test_acc': []}
    record['Cs_test_acc'].append(top1_source.avg)
    record['Ct_test_acc'].append(top1_target.avg)
    record['CST_test_acc'].append(top1_st.avg)
    record['diff_pred_same'].append(pred_diff.avg)
    record['diff_prob_norm'].append(prob_diff.avg)

    filename = 'ncf_'+str(args.nspc)+'_' + args.src+'2'+args.tar+'_'+str(args.shot)+'shot_log.txt'
    log = open(os.path.join(args.log, filename), 'a')
    log.write("\n")
    log.write(" Test:epoch: %d, Loss CS: %4f, Loss CT: %4f, Top1CS: %3f, Top1CT: %3f, Top1CST: %3f !!!" %\
              (epoch, losses_source.avg, losses_target.avg, top1_source.avg, top1_target.avg, top1_st.avg))
    log.close()

    return max(top1_source.avg, top1_target.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # print("meter: ", self.val, self.avg, val, n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print("meter after: ", self.val, self.avg, self.sum, self.count)


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    ## annealing strategy 1
    # epoch_total = int(args.epochs / args.test_freq)
    # epoch = int((epoch + 1) / args.test_freq)
    lr = args.lr / pow((1 + 10 * epoch / args.epochs), 0.75)
    lr_pretrain = args.lr * 0.1 / pow((1 + 10 * epoch / args.epochs), 0.75) # 0.001 / pow((1 + 10 * epoch / epoch_total), 0.75)
    ## annealing strategy 2
    # exp = epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
    # lr = args.lr * (args.gamma ** exp)
    # lr_pretrain = lr * 0.1 #1e-3
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            param_group['lr'] = lr_pretrain
        else:
            param_group['lr'] = lr




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output.shape: bs x n_cls
    # target.shape: bs
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() # bs x topk --> topk x bs, select the top-k prediction indeces
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # 1 x bs --> topk x bs
    # print("accuracy: ", pred.shape, output.shape, correct.shape)

    res = []
    for k in topk: # (1, 5)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
