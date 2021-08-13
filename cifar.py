# Code is adapted from "https://github.com/google-research/augmix/blob/master/cifar.py",
# which is originally licensed under Apache 2.0.

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
import torch.optim as optim
import time

from utils import get_log_dir_path, AverageMeter, save_checkpoint, AugMixDataset

# from augmix_utils.dataset import AugMixDataset
from models.cifar.wideresnet_cnsn import WideResNet
from models.cifar.allconv_cnsn import AllConvNet
from models.cifar.resnext_cnsn import resnext29
from models.cifar.densenet_cnsn import densenet

import argparse

parser = argparse.ArgumentParser(description='crossnorm and selfnorm for'
                                             'robust CIFAR-10 and CIFAR-100 training.')
parser.add_argument('--model', default=None, type=str,
                    help='model type')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                   help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10/cifar100)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print every n iterations')
parser.add_argument('--exp_dir', default='./exp', type=str,
                    help='exp dir')
parser.add_argument('--data_dir', default='./data', type=str,
                    help='data dir')
parser.add_argument('--corrupt_data_dir', default=None, type=str,
                    help='corruption data dir')
parser.add_argument('--exp_id', default='cnsn-wrn-cifar', type=str,
                    help='exp id')
parser.add_argument('--resume', default=None, type=str,
                    help='resume from checkpoint')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate or not')
parser.add_argument('--cn_prob', default=None, type=float,
                    help='crossnorm probability')
parser.add_argument('--active_num', default=None, type=int,
                    help='active crossnorm num')
parser.add_argument('--pos', default=None, type=str,
                    help='position of cnsn inside a residual module')
parser.add_argument('--beta', default=None, type=float,
                    help='beta distribution to sample the'
                         ' ratio of a cropping bbx for crossnorm')
parser.add_argument('--crop', default=None, type=str,
                    help='crop a bbx in 2-instance crossnorm: '
                         'neither/style/content/both')
parser.add_argument('--cnsn_type', default=None, type=str,
                    help='sn/cn/cnsn, type of using selfnorm and crossnorm')
parser.add_argument('--consist_wt', default=None, type=float,
                    help='weight for the consistency regularization term')


parser.set_defaults(verbose=True)

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def train(net, train_loader, optimizer, scheduler):
    """Train for one epoch."""
    print('running train')
    losses = AverageMeter()
    net.train()

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = net(inputs)
        loss = F.cross_entropy(logits, targets)
        losses.update(loss.item(), inputs.size(0))

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % args.print_freq == 0:
            print('Train Loss {:.3f}'.format(losses.avg))

    return losses.avg


def train_cn(net, train_loader, optimizer, scheduler):
    """Train for one epoch."""
    print('running train_cn')
    losses = AverageMeter()
    net.train()

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # crossnorm or not
        r = np.random.rand(1)
        if r < args.cn_prob:
            logits = net(inputs, aug=True)
        else:
            logits = net(inputs, aug=False)

        loss = F.cross_entropy(logits, targets)
        losses.update(loss.item(), inputs.size(0))

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % args.print_freq == 0:
            print('Train Loss {:.3f}'.format(losses.avg))

    return losses.avg


def train_cn_consistency(net, train_loader, optimizer, scheduler):
    """Train for one epoch."""
    print('running train_cn_consistency')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    s_losses = AverageMeter()
    c_losses = AverageMeter()
    net.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        r = np.random.rand(1)
        if r < args.cn_prob:
            logits_clean = net(input, aug=False)
            # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(logits_clean, target)

            logits_aug1 = net(input, aug=True)
            logits_aug2 = net(input, aug=True)

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                logits_aug1, dim=1), F.softmax(
                logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            consist_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

            s_losses.update(loss.item(), input.size(0))
            c_losses.update(consist_loss.item(), input.size(0))
            loss += args.consist_wt * consist_loss
            losses.update(loss.item(), input.size(0))
        else:
            output = net(input, aug=False)
            loss = F.cross_entropy(output, target)
            s_losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Supervised Loss {s_losses.val:.4f} ({s_losses.avg:.4f})\t'
                  'Consistency Loss {c_losses.val:.4f} ({c_losses.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    s_losses=s_losses, c_losses=c_losses, loss=losses))

    return losses.avg


def train_cn_augmix(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  print('running train_cn_augmix')
  s_losses = AverageMeter()
  c_losses = AverageMeter()
  losses = AverageMeter()
  net.train()
  loss_ema = 0.
  for i, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    # print('augmix forward...')
    images_all = torch.cat(images, 0).cuda()
    targets = targets.cuda()
    logits_all = net(images_all, aug=False)

    logits_clean, logits_aug1, logits_aug2 = torch.split(
        logits_all, images[0].size(0))

    # Cross-entropy is only computed on clean images
    loss = F.cross_entropy(logits_clean, targets)

    p_clean, p_aug1, p_aug2 = F.softmax(
        logits_clean, dim=1), F.softmax(
        logits_aug1, dim=1), F.softmax(
        logits_aug2, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    consist_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    s_losses.update(loss.item(), images[0].size(0))
    c_losses.update(consist_loss.item(), images[0].size(0))
    loss += 12 * consist_loss
    losses.update(loss.item(), images[0].size(0))

    r = np.random.rand(1)
    if r < args.cn_prob:
        logits_cn_aug1 = net(images[0], aug=True)
        logits_cn_aug2 = net(images[0], aug=True)
        #
        p_cn_aug1, p_cn_aug2 = F.softmax(
            logits_cn_aug1, dim=1), F.softmax(
            logits_cn_aug2, dim=1)
        p_cn_mixture = torch.clamp((p_clean + p_cn_aug1 + p_cn_aug2) / 3., 1e-7, 1).log()
        cn_consist_loss = (F.kl_div(p_cn_mixture, p_clean, reduction='batchmean') +
                           F.kl_div(p_cn_mixture, p_cn_aug1, reduction='batchmean') +
                           F.kl_div(p_cn_mixture, p_cn_aug2, reduction='batchmean')) / 3.
        loss += args.consist_wt * cn_consist_loss

    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    if i % args.print_freq == 0:
      print('Supervised Loss {s_losses.val:.4f} ({s_losses.avg:.4f})\t'
            'Consistency Loss {c_losses.val:.4f} ({c_losses.avg:.4f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             s_losses=s_losses, c_losses=c_losses, loss=losses))

  return loss_ema


def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(os.path.join(base_path, corruption + '.npy'))
    test_data.targets = torch.LongTensor(np.load(os.path.join(base_path, 'labels.npy')))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1000,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs)


def main():
    torch.manual_seed(1)
    np.random.seed(1)

    # datasets
    if 'augmix' in args.exp_id:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4)])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = preprocess

    if args.dataset.lower() == 'cifar-10':
        print('using cifar-10 data ...')
        train_data = datasets.CIFAR10(
            root=args.data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            root=args.data_dir, train=False, transform=test_transform, download=True)
        base_c_path = args.corrupt_data_dir
        num_classes = 10
    elif args.dataset.lower() == 'cifar-100':
        print('using cifar-100 data ...')
        train_data = datasets.CIFAR100(
            root=args.data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            root=args.data_dir, train=False, transform=test_transform, download=True)
        base_c_path = args.corrupt_data_dir
        num_classes = 100
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    assert os.path.isdir(base_c_path)
    if 'augmix' in args.exp_id:
        train_data = AugMixDataset(train_data, preprocess, all_ops=False, mixture_width=3,
                                   mixture_depth=-1, aug_severity=3, no_jsd=False, image_size=32)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1000,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # model
    print('model: {}'.format(args.model))
    if args.model == 'wideresnet':
        net = WideResNet(40, num_classes=num_classes, widen_factor=2, drop_rate=0,
                         active_num=args.active_num, pos=args.pos,
                         beta=args.beta, crop=args.crop, cnsn_type=args.cnsn_type)
    elif args.model == 'allconv':
        net = AllConvNet(num_classes, active_num=args.active_num, pos=args.pos,
                         beta=args.beta, crop=args.crop,
                         cnsn_type=args.cnsn_type)
    elif args.model == 'resnext':
        net = resnext29(num_classes=num_classes, config=args)
    elif args.model == 'densenet':
        net = densenet(num_classes=num_classes, config=args)
    else:
        raise Exception('unkown model: {}'.format(args.model))

    para_num = sum(p.numel() for p in net.parameters())
    print('model param #: {}'.format(para_num))

    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    # optimizer
    optimizer = optim.SGD(net.parameters(), args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=True)
    for group in optimizer.param_groups:
        print('lr: {}, weight_decay: {}, momentum: {}, nesterov: {}'
              .format(group['lr'], group['weight_decay'], group['momentum'], group['nesterov']))

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.lr))

    if args.resume:
        # print_logits(net, train_loader, 100)
        print('resume checkpoint: {}'.format(args.resume))
        exp_dir_idx = args.resume.rindex('/')
        exp_dir = args.resume[:exp_dir_idx]
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            # print('exp_dir: {}'.format(exp_dir))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0
        best_acc = 0.
        exp_dir = get_log_dir_path(args.exp_dir, args.exp_id)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

    if args.evaluate:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_acc = test(net, test_loader)
        print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            test_loss, 100 - 100. * test_acc))

        test_c_acc = test_c(net, test_data, base_c_path)
        print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
        return

    print('exp_dir: {}'.format(exp_dir))
    log_file = os.path.join(exp_dir, 'log.txt')
    names = ['epoch', 'lr', 'Train Loss', 'Test Err1' 'Best Test Err1']
    with open(log_file, 'a') as f:
        f.write('dataset: {}\n'.format(args.dataset))
        f.write('batch size: {}\n'.format(args.batch_size))
        f.write('lr: {}\n'.format(args.lr))
        f.write('momentum: {}\n'.format(args.momentum))
        f.write('weight_decay: {}\n'.format(args.weight_decay))
        for per_name in names:
            f.write(per_name + '\t')
        f.write('\n')
    # print('=> Training the base model')
    print('start_epoch {}'.format(start_epoch))
    print('total epochs: {}'.format(args.epochs))
    print('best_acc: {}'.format(best_acc))
    # print('best_err5: {}'.format(best_err5))

    if args.cn_prob:
        print('cn_prob: {}'.format(args.cn_prob))
    if args.consist_wt:
        print('consist_wt: {}'.format(args.consist_wt))
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']

        if 'augmix' in args.exp_id and 'cn' in args.cnsn_type:
            assert args.cn_prob > 0 and args.consist_wt > 0
            train_loss_ema = train_cn_augmix(net, train_loader, optimizer, scheduler)
        elif 'consist' in args.exp_id and 'cn' in args.cnsn_type:
            assert args.cn_prob > 0 and args.consist_wt > 0
            train_loss_ema = train_cn_consistency(net, train_loader, optimizer, scheduler)
        elif 'cn' in args.cnsn_type:
            assert args.cn_prob > 0
            train_loss_ema = train_cn(net, train_loader, optimizer, scheduler)
        else:
            train_loss_ema = train(net, train_loader, optimizer, scheduler)

        test_loss, test_acc = test(net, test_loader)
        # test_c_acc = test_c(net, test_data, base_c_path)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        save_checkpoint(net, {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, exp_dir, epoch=None)

        values = [train_loss_ema, 100 - 100. * test_acc, 100 - 100. * best_acc]
        with open(log_file, 'a') as f:
            f.write('{:d}\t'.format(epoch))
            f.write('{:g}\t'.format(lr))
            for per_value in values:
                f.write('{:2.2f}\t'.format(per_value))
            f.write('\n')
        print('exp_dir: {}'.format(exp_dir))

    test_c_acc = test_c(net, test_data, base_c_path)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
    with open(log_file, 'a') as f:
        f.write('{:2.2f}\t'.format(100 - 100. * test_c_acc))
        f.write('\n')


if __name__ == '__main__':
    main()