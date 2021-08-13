
from __future__ import print_function
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
# from augmix_utils.dataset import AugMixDataset
from models.imagenet.resnet_cnsn import resnet50
from models.imagenet.resnet_ibn_cnsn import resnet50_ibn_a, resnet50_ibn_b
from utils import get_log_dir_path, AverageMeter, save_checkpoint, AugMixDataset
from models.cnsn import cn_op_2ins_space_chan

import argparse

parser = argparse.ArgumentParser(description='crossnorm and selfnorm for'
                                             'robust ImageNet training.')
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
parser.add_argument('--pretrained', default=None, type=str,
                    help='pretrained model path')
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

# Raw AlexNet errors taken from https://github.com/hendrycks/robustness
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]


def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR (linearly scaled to batch size) decayed by 10 every n / 3 epochs."""
  b = args.batch_size / 256.
  k = args.epochs // 3
  if epoch < k:
    m = 1
  elif epoch < 2 * k:
    m = 0.1
  else:
    m = 0.01
  lr = args.lr * m * b
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def error(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


def compute_mce(corruption_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  ce_dict = {}
  for i in range(len(CORRUPTIONS)):
    avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
    ce = 100 * avg_err / ALEXNET_ERR[i]
    ce_dict[CORRUPTIONS[i]] = ce
    mce += ce / 15
  return mce, ce_dict


def print_ces(ce_dict):
    print('individual CEs: ')
    for per in CORRUPTIONS:
        print('{0}: {ce: .2f}'.format(per, ce=ce_dict[per]))


def train(model, train_loader, optimizer):
    """Train for one epoch."""
    print('running train')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        # print('\nbasic training...')
        output = model(input)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        err1, err5 = error(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose is True:
            # print('Train Loss {:.3f}'.format(losses.avg))
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top1 err {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Top5 err {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                   i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            # print(target[:10])
            # exit()

    return top1.avg


def train_cn_image(model, train_loader, optimizer):
    """Train for one epoch."""
    print('running train_cn_image')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        r = np.random.rand(1)
        # print('random prob: {:2f}'.format(r[0]))
        if r < args.cn_prob:
            input = cn_op_2ins_space_chan(input, beta=args.beta, crop=args.crop)

        output = model(input, aug=False)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        err1, err5 = error(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose is True:
            # print('Train Loss {:.3f}'.format(losses.avg))
            print('Epoch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top1 err {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Top5 err {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                   i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            # print(target[:10])
            # exit()
        # if i == 10:
        #     break

    return top1.avg


def train_cn_image_consist(model, train_loader, optimizer):
    """Train for one epoch."""
    print('running train_cn_image_consist')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    s_losses = AverageMeter()
    c_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    # make sure using crop because the two image augmentations should be different
    assert args.beta is not None
    assert args.crop in ['both', 'style', 'content']
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        r = np.random.rand(1)
        if r < args.cn_prob:
            # print('\ncross norm training...')
            # print('computing logits_clean')
            logits_clean = model(input, aug=False)
            # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(logits_clean, target)

            # # print('computing logits_aug1')
            input_aug1 = cn_op_2ins_space_chan(input, beta=args.beta, crop=args.crop)
            logits_aug1 = model(input_aug1, aug=False)

            # # print('computing logits_aug2')
            input_aug2 = cn_op_2ins_space_chan(input, beta=args.beta, crop=args.crop)
            logits_aug2 = model(input_aug2, aug=False)
            #
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
            # print('\nbasic training...')
            logits_clean = model(input, aug=False)
            loss = F.cross_entropy(logits_clean, target)
            s_losses.update(loss.item(), input.size(0))

        # measure accuracy and record loss
        err1, err5 = error(logits_clean, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # print('Train Loss {:.3f}'.format(loss_ema))
            print('Iter: [{0}/{1}]\t'
                  'Supervised Loss {s_losses.val:.4f} ({s_losses.avg:.4f})\t'
                  'Consistency Loss {c_losses.val:.4f} ({c_losses.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(train_loader),
                   s_losses=s_losses, c_losses=c_losses, loss=losses))

    return top1.avg


def train_cn_image_augmix(net, train_loader, optimizer):
  """Train for one epoch."""
  print('running train_cn_image_augmix')
  net.train()
  losses = AverageMeter()
  s_losses = AverageMeter()
  c_losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  end = time.time()
  for i, (images, targets) in enumerate(train_loader):
    # Compute data loading time
    # data_time = time.time() - end

    images_all = torch.cat(images, 0)
    images_all = images_all.cuda()
    targets = targets.cuda()

    r = np.random.rand(1)
    if r < args.cn_prob:
        images_all = cn_op_2ins_space_chan(images_all, beta=args.beta, crop=args.crop)

    logits_all = net(images_all)
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

    err1, err5 = error(logits_clean, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
    top1.update(err1.item(), images[0].size(0))
    top5.update(err5.item(), images[0].size(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute batch computation time and update moving averages.
    batch_time = time.time() - end
    end = time.time()

    if i % args.print_freq == 0:
        # print('Train Loss {:.3f}'.format(loss_ema))
        print('Iter: [{0}/{1}]\t'
              'Supervised Loss {s_losses.val:.4f} ({s_losses.avg:.4f})\t'
              'Consistency Loss {c_losses.val:.4f} ({c_losses.avg:.4f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(train_loader),
               s_losses=s_losses, c_losses=c_losses, loss=losses))

    # if i == 10:
    #     break

  return top1.avg


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


def test_c(net, test_transform):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = {}
  for c in CORRUPTIONS:
    print(c)
    for s in range(1, 6):
      valdir = os.path.join(args.corrupt_data_dir, c, str(s))
      test_c_dataset = datasets.ImageFolder(valdir, test_transform)
      val_loader = torch.utils.data.DataLoader(
          test_c_dataset,
          batch_size=1000,
          shuffle=False,
          num_workers=args.workers,
          pin_memory=True)

      loss, acc1 = test(net, val_loader)
      if c in corruption_accs:
        corruption_accs[c].append(acc1)
      else:
        corruption_accs[c] = [acc1]

      print('\ts={}: Test Loss {:.3f} | Test Acc1 {:.3f}'.format(
          s, loss, 100. * acc1))

  return corruption_accs


def main():
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean, std)])
  if 'augmix' in args.exp_id:
      print('using augmix data preprocessing...')
      train_transform = transforms.Compose(
          [transforms.RandomResizedCrop(224),
           transforms.RandomHorizontalFlip()])
  else:
      print('using only standard data preprocessing...')
      train_transform = transforms.Compose(
          [transforms.RandomResizedCrop(224),
           transforms.RandomHorizontalFlip(),
           preprocess])

  test_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      preprocess,
  ])

  traindir = os.path.join(args.data_dir, 'train')
  valdir = os.path.join(args.data_dir, 'validation')
  train_dataset = datasets.ImageFolder(traindir, train_transform)

  assert os.path.isdir(args.corrupt_data_dir)
  if 'augmix' in args.exp_id:
    train_dataset = AugMixDataset(train_dataset, preprocess, all_ops=False, mixture_width=3,
                                  mixture_depth=-1, aug_severity=1, no_jsd=False, image_size=224)
  # print('batch_size: {}'.format(args.batch_size))
  # print('workers: {}'.format(args.workers))
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.workers,
      pin_memory=True)
  test_dataset = datasets.ImageFolder(valdir, test_transform)

  val_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=1000,
      shuffle=False,
      num_workers=args.workers,
      pin_memory=True)

  print('model: {}'.format(args.model))
  if args.model == 'resnet50':
      net = resnet50(args)
  elif args.model == 'resnet50_ibn_a':
      net = resnet50_ibn_a(args)
  elif args.model == 'resnet50_ibn_b':
      net = resnet50_ibn_b(args)

  para_num = sum(p.numel() for p in net.parameters())
  print('model param #: {}'.format(para_num))
  # exit()

  if args.pretrained:
      print('pretrained model: {}'.format(args.pretrained))
      state_dict = torch.load(args.pretrained)
      net.load_state_dict(state_dict, strict=False)

  print('optimizer momentum: {}'.format(args.momentum))
  print('optimizer weight_decay: {}'.format(args.weight_decay))

  optimizer = torch.optim.SGD(
      net.parameters(),
      args.lr,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0

  if args.resume:
      # print('resume checkpoint: {}'.format(args.resume))
      exp_dir_idx = args.resume.rindex('/')
      exp_dir = args.resume[:exp_dir_idx]
      if os.path.isfile(args.resume):
          print("=> loading checkpoint '{}'".format(args.resume))
          checkpoint = torch.load(args.resume)
          start_epoch = checkpoint['epoch']
          best_acc = checkpoint['best_acc']
          net.load_state_dict(checkpoint['state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          # print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
          # print('exp_dir: {}'.format(exp_dir))
      else:
          print("=> no checkpoint found at '{}'".format(args.resume))
      # best_val_acc, test_acc, start_epoch = \
      #     utils.load_checkpoint(args, model, optimizer)

  else:
      start_epoch = 0
      best_acc = 0.
      exp_dir = get_log_dir_path(args.exp_dir, args.exp_id)
      if not os.path.exists(exp_dir):
          os.makedirs(exp_dir)

  if args.evaluate:
    test_loss, test_acc1 = test(net, val_loader)
    print('Clean\n\tTest Loss {:.3f} | Test Acc1 {:.3f}'.format(
        test_loss, 100 * test_acc1))
    # exit()
    corruption_accs = test_c(net, test_transform)
    for c in CORRUPTIONS:
        print('\t'.join(map(str, [c] + corruption_accs[c])))

    mce, ce_dict = compute_mce(corruption_accs)
    print_ces(ce_dict)
    print('mCE (normalized by AlexNet): ', mce)
    return

  print('exp_dir: {}'.format(exp_dir))
  log_file = os.path.join(exp_dir, 'log.txt')
  names = ['epoch', 'lr', 'Train Err1', 'Test Err1' 'Best Test Err1']
  with open(log_file, 'a') as f:
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
  print('Beginning training from epoch:', start_epoch)

  if args.cn_prob:
      print('cn_prob: {}'.format(args.cn_prob))
  if args.consist_wt:
      print('consist_wt: {}'.format(args.consist_wt))

  for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)
    lr = optimizer.param_groups[0]['lr']
    print('lr: {}'.format(lr))

    if 'augmix' in args.exp_id:  # for CrossNorm in image space, 'cn' is not in cnsn_type
        assert args.cn_prob > 0
        train_err1 = train_cn_image_augmix(net, train_loader, optimizer)
    elif 'consist' in args.exp_id:  # for CrossNorm in image space, 'cn' is not in cnsn_type
        assert args.cn_prob > 0
        train_err1 = train_cn_image_consist(net, train_loader, optimizer)
    elif 'cn' in args.exp_id:  # for CrossNorm in image space, 'cn' is not in cnsn_type
        assert args.cn_prob > 0
        train_err1 = train_cn_image(net, train_loader, optimizer)
    else:
        train_err1 = train(net, train_loader, optimizer)

    test_loss, test_acc = test(net, val_loader)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)

    save_checkpoint(net, {
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, exp_dir, epoch=epoch)

    values = [train_err1, 100 - 100. * test_acc, 100 - 100. * best_acc]
    with open(log_file, 'a') as f:
        f.write('{:d}\t'.format(epoch))
        f.write('{:g}\t'.format(lr))
        for per_value in values:
            f.write('{:2.2f}\t'.format(per_value))
        f.write('\n')
    print('exp_dir: {}'.format(exp_dir))

  corruption_accs = test_c(net, test_transform)
  for c in CORRUPTIONS:
    print('\t'.join(map(str, [c] + corruption_accs[c])))

  mce, ce_dict = compute_mce(corruption_accs)
  print_ces(ce_dict)
  print('mCE (normalized by AlexNet): {:.2f}'.format(mce))
  with open(log_file, 'a') as f:
    f.write('individual corruption errors: \n')
    for per in CORRUPTIONS:
        f.write('{0}: {ce:.2f}\n'.format(per, ce=ce_dict[per]))
    f.write('mCE: {:.2f}\t'.format(mce))
    f.write('\n')


if __name__ == '__main__':
    main()