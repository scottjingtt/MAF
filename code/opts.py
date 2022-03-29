import argparse


def opts():
    parser = argparse.ArgumentParser(description='Train alexnet on the cub200 dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path_source', type=str, default='',
                        help='Root of train data set of the source domain')

    parser.add_argument('--data_path_target_tr', type=str, default='',
                        help='Root of train data set of the target domain')
    parser.add_argument('--data_path_target', type=str, default='',
                        help='Root of the test data set of the target domain')
    parser.add_argument('--src', type=str, default='amazon',
                        help='choose between amazon | dslr | webcam')
    parser.add_argument('--tar_tr', type=str, default='webcam',
                        help='choose between amazon | dslr | webcam')
    parser.add_argument('--tar', type=str, default='webcam',
                        help='choose between amazon | dslr | webcam')
    parser.add_argument('--num_classes', type=int, default=31,
                        help='number of classes of data used to fine-tune the pre-trained model')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size_s', '-b-s', type=int, default=16, help='Batch size of the source data.')
    parser.add_argument('--batch_size_t', '-b-t', type=int, default=16, help='Batch size of the target data.')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[49, 79, 119],
                        help='Decrease learning rate at these epochs[used in step decay].')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # checkpoints
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='Checkpoints path to resume(default none)')
    parser.add_argument('--pretrained_checkpoint', type=str, default='', help='Self-Pretrained checkpoint to resume (default none)')
    parser.add_argument('--test_only', '-t', action='store_true', help='Test only flag')
    # Architecture
    parser.add_argument('--arch', type=str, default='resnet50', help='Model name')
    parser.add_argument('--flag', type=str, default='original', help='flag for different settings')
    parser.add_argument('--pretrained', action='store_true', help='whether using pretrained model')
    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--test_freq', default=10, type=int,
                        help='test frequency (default: 1)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--score_frep', default=300, type=int,
                        metavar='N', help='print frequency (default: 300, not download score)')

    parser.add_argument('--way', default=31, type=int,
                        metavar='N', help='way')
    parser.add_argument('--nspc', default=20, type=int,
                        metavar='N', help='n samples of source per class')
    parser.add_argument('--shot', default=3, type=int,
                        metavar='N', help='shot')
    parser.add_argument('--dataset', default='Office', type=str,
                        help='dataset name')
    parser.add_argument('--split', default=0, type=int,
                        help='id of random index split for target few shot')
    parser.add_argument('--mode', default='both', type=str,
                        help='which augmentation loss will use')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='Optimizer: Adam or SGD')
    parser.add_argument('--param2', default=0, type=float,
                        help='weight hyper-param')
    parser.add_argument('--bestAcc', default=0.0, type=float, help='Best test accuracy')
    args = parser.parse_args() # this will read bash variables/arguments

    args.log = args.log + '_' + args.src + '2' + args.tar + '_' + args.arch + '_' + args.flag
    print("data_path_source: ", args.data_path_source, args.arch)
    return args
