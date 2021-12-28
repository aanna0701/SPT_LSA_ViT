from utils.autoaug import SVHNPolicy
from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from colorama import Fore, Style
from torchsummary import summary
from utils.losses import LabelSmoothingCrossEntropy
import os
from utils.sampler import RASampler
# import models.create_model as m
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from models.vit_pytorch.git import *
from utils.scheduler import build_scheduler
from utils.Regularizations import CosineSimiliarity, Identity
from utils.throughput import throughput
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0
input_size = 32
MODELS = ['vit', 'lovit', 'swin', 'g-vit','g-vit2','g-vit3', 'pit', 'cait', 't2t', 'cvt', 'deepvit',
          'resnet', 'resnet110','effinet', 'effiB7']


def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args
    parser.add_argument('--data_path', default='./dataset', type=str, help='dataset path')
    
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'M-IMNET', 'SVHN', 'IMNET'], type=str, help='Image Net dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    
    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--model', type=str, default='deit', choices=MODELS)

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_true', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag', default='')

    parser.add_argument('--seed', type=int, help='seed')

    parser.add_argument('--down_conv', action='store_true', help='down conv embedding')
    
    parser.add_argument('--sd', default=0, type=float, help='rate of stochastic depth')
    
    parser.add_argument('--ver', default=1, type=int, help='Version')
    
    parser.add_argument('--resume', default=False, help='Version')
    
    # Augmentation parameters
    parser.add_argument('--aa', action='store_true', help='Auto augmentation used'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--n_trans', type=int, default=4, help='The num of trans')
    parser.add_argument('--gam', default=0, type=float, help='Regularizaer')
    parser.add_argument('--lam', default=0, type=float, help='hyperparameter of similiarity loss')
    parser.add_argument('--init_type', default='aistats', choices=['aistats', 'identity'])
    parser.add_argument('--scale', default=0, type=float, help='init noise')

    parser.add_argument('--merging_size', default=2, type=int)
    parser.add_argument('--pe_dim', default=128, type=int)
    parser.add_argument('--is_base', action='store_true')
    parser.add_argument('--is_coord', action='store_true')
    parser.add_argument('--is_rpe', action='store_true')
    parser.add_argument('--is_ape', action='store_true')
    parser.add_argument('--is_LSA', action='store_true')
    parser.add_argument('--STT_head', default=16, type=int)
    parser.add_argument('--STT_depth', default=1, type=int)
    
    
    # Mixup params
  
    parser.add_argument('--cm',action='store_true' , help='Use Cutmix')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    parser.add_argument('--mu',action='store_true' , help='Use Mixup')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    # Autoaugmentation
    parser.add_argument('--rand_aug', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    
    parser.add_argument('--enable_rand_aug', action='store_true', help='Enabling randaugment')
    
    parser.add_argument('--enable_deit', action='store_true', help='Enabling randaugment')
    parser.add_argument('--dropout', type=float, help='dropout rate')
    parser.add_argument('--ra', type=int, default=0, help='repeated augmentation')
    
    # Random Erasing
    parser.add_argument('--re', default=0, type=float, help='Random Erasing probability')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')

    return parser


def main(args):
    global best_acc1    
    
    torch.cuda.set_device(args.gpu)

    '''
        Dataset
    '''
    if args.dataset == 'CIFAR10':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR10')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32
        patch_size = 4
        in_channels = 3
        
    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32
        patch_size = 4
        in_channels = 3
        
    elif args.dataset == 'SVHN':
        print(Fore.YELLOW+'*'*80)
        logger.debug('SVHN')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970) 
        img_size = 32
        patch_size = 4
        in_channels = 3
        
    elif args.dataset == 'IMNET':
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        print(Fore.YELLOW+'*'*80)
        logger.debug('IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 1000
        img_mean, img_std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        img_size = 224
        patch_size = 16
        in_channels = 3
        
    elif args.dataset == 'T-IMNET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('T-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64
        patch_size = 8
        in_channels = 3
        
    
    '''
        Model 
    '''    
    
    # ViTs
    
    if not args.is_rpe and not args.is_coord:
        args.is_ape = True
            
    dropout = False
    if args.dropout:
        dropout = args.dropout
    if args.model == 'vit':
        from models.vit_pytorch.vit import ViT        
        dim_head = args.channel // args.heads
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, 
                    mlp_dim_ratio=2, depth=args.depth, heads=args.heads, dim_head=dim_head, pe_dim=args.pe_dim,
                    dropout=dropout, stochastic_depth=args.sd, is_base=args.is_base, eps=args.scale, merging_size=args.merging_size,
                    is_coord=args.is_coord, is_LSA=args.is_LSA, n_trans=args.n_trans,
                    STT_head=args.STT_head, STT_depth=args.STT_depth, is_rpe=args.is_rpe, is_ape=args.is_ape)

        # (n_trans=args.n_trans, is_base=False, is_learn=args.is_trans_learn, init_noise = args.init_type, eps=args.scale, 
        # padding_mode=args.padding, type_trans=args.type_trans, n_token=args.n_token,
        # img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, mlp_dim_ratio=2, depth=args.depth, heads=args.heads, dim_head=dim_head, dropout=dropout, stochastic_depth=args.sd)
   
        
    elif args.model == 'cait':
        from models.vit_pytorch.cait import CaiT        
        dim_head = args.channel // args.heads
        if img_size == 64:
            patch_size = 8
        elif img_size == 32:
            patch_size = 4
        else:
            patch_size = 16
            
        model = CaiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, stochastic_depth=args.sd, is_LSA=args.is_LSA, is_SPT=(not args.is_base))
    
    
    elif args.model == 'pit':
        from models.vit_pytorch.pit import PiT
        if img_size == 32:
            patch_size = 2
        elif img_size > 32:
            patch_size = 4
        

        args.channel = 96
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        
        dim_head = args.channel // args.heads[0]
        
        model = PiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, 
                    mlp_dim_ratio=2, depth=args.depth, heads=args.heads, dim_head=dim_head, dropout=dropout, 
                    stochastic_depth=args.sd, is_base=args.is_base,  merging_size=args.merging_size, pe_dim=args.pe_dim,
                    eps=args.scale, is_coord=args.is_coord, is_LSA=args.is_LSA, 
                    n_trans=args.n_trans, is_rpe=args.is_rpe, is_ape=args.is_ape)


    elif args.model =='t2t':
        from models.vit_pytorch.t2t import T2T_ViT
            
        model = T2T_ViT(img_size=img_size, num_classes=n_classes, drop_path_rate=args.sd)
        
    elif args.model =='cvt':
        from models.vit_pytorch.cvt import CvT
        if img_size == 32:
            patch_size = 3
        else:
            patch_size = 7
        model = CvT(num_classes=n_classes, patch_size=patch_size, stochastic_depth=args.sd)
        
    elif args.model =='swin':
        from models.vit_pytorch.swin import SwinTransformer
        if img_size > 64:
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            mlp_ratio = 4
            window_size = 7
            patch_size = 4
        else:
            depths = [2, 6, 4]
            num_heads = [3, 6, 12]
            mlp_ratio = 2
            window_size = 4
            patch_size //= 2
            
        model = SwinTransformer(n_trans=args.n_trans, img_size=img_size, window_size=window_size, drop_path_rate=args.sd, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                is_base=args.is_base,  merging_size=args.merging_size, eps=args.scale,  pe_dim=args.pe_dim, 
                                is_coord=args.is_coord, is_LSA=args.is_LSA, is_rpe=args.is_rpe, is_ape=args.is_ape
                                )
   
    elif args.model =='resnet':
        from models.conv_cifar_pytoch.resnet import resnet56
        
        model = resnet56(num_classes=n_classes)
   
    elif args.model =='resnet110':
        from models.conv_cifar_pytoch.resnet import resnet110
        
        model = resnet110(num_classes=n_classes)
        
        
    elif args.model == 'effinet':
        from models.conv_cifar_pytoch.efficientnet import EfficientNetB0
        
        model = EfficientNetB0(num_classes=n_classes)
        
        
    elif args.model == 'effiB7':
        from models.efficientnet_pytorch.effib7 import EfficientNet
        
        model = EfficientNet.from_name(model_name='efficientnet-b7', image_size=img_size , num_classes=n_classes)
        
    
    model.cuda(args.gpu)  
        
    print(Fore.GREEN+'*'*80)
    logger.debug(f"Creating model: {model_name}")    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f'Number of params: {format(n_parameters, ",")}')
    logger.debug(f'FLOPs: {format(model.flops(), ",")}')
    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    print('*'*80+Style.RESET_ALL)
    
    '''
        Criterion
    '''
    
    if args.ls:
        print(Fore.YELLOW + '*'*80)
        logger.debug('label smoothing used')
        print('*'*80+Style.RESET_ALL)
        criterion = LabelSmoothingCrossEntropy()
    
    else:
        criterion = nn.CrossEntropyLoss()
    
        
    if args.sd > 0.:
        print(Fore.YELLOW + '*'*80)
        logger.debug(f'Stochastic depth({args.sd}) used ')
        print('*'*80+Style.RESET_ALL)         



    criterion = criterion.cuda(args.gpu)

    
    '''
        Trainer
    '''

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]


    if args.cm:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Cutmix used')
        print('*'*80 + Style.RESET_ALL)
    if args.mu:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Mixup used')
        print('*'*80 + Style.RESET_ALL)
    if args.ra > 1:        
        
        print(Fore.YELLOW+'*'*80)
        logger.debug(f'Repeated Aug({args.ra}) used')
        print('*'*80 + Style.RESET_ALL)

    '''
        Data Augmentation
    '''
    augmentations = []
    
    if args.aa == True:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Autoaugmentation used')      
        
        if 'CIFAR' in args.dataset:
            print("CIFAR Policy")
            from utils.autoaug import CIFAR10Policy
            augmentations += [
                
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy()
            ]
            
        elif 'SVHN' in args.dataset:
            print("SVHN Policy")    
            from utils.autoaug import SVHNPolicy
            augmentations += [
                
              transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                SVHNPolicy()
            ]
        
            
        elif args.dataset == 'IMNET':
            print("ImageNet Policy")    
            from utils.autoaug import ImageNetPolicy
            augmentations += [
                transforms.RandomResizedCrop(224),
                ImageNetPolicy()
            ]
            
        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [                
              transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy()
            ]
            
        print('*'*80 + Style.RESET_ALL)
        

    if args.re > 0:
        from utils.random_erasing import RandomErasing
        print(Fore.YELLOW + '*'*80)
        logger.debug(f'Random erasing({args.re}) used ')
        print('*'*80+Style.RESET_ALL)    
        
        
        augmentations += [                
            transforms.ToTensor(),
            *normalize,
            RandomErasing(probability = args.re, sh = args.re_sh, r1 = args.re_r1, mean=img_mean)]
    
    else:
        augmentations += [                
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            *normalize]
    
    
    augmentations = transforms.Compose(augmentations)

    '''
        Data Loader
    '''
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'CIFAR100':

        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
        
    elif args.dataset == 'SVHN':

        train_dataset = datasets.SVHN(
            root=args.data_path, split='train', download=True, transform=augmentations)
        val_dataset = datasets.SVHN(
            root=args.data_path, split='test', download=True, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'IMNET':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'imnet', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'imnet', 'val'), 
            transform=transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.CenterCrop(img_size), transforms.ToTensor(), *normalize]))
        
    elif args.dataset == 'T-IMNET':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'val'), 
            transform=transforms.Compose([
            transforms.Resize(img_size), transforms.ToTensor(), *normalize]))
      

    train_loader = torch.utils.data.DataLoader(
        train_dataset,  num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    '''
        Training
    '''
    
    # no_wd = list()
    # yes_wd = list()
    
    # all_params = set(model.parameters())
    # no_wd = set()
    # for m in list(model.parameters()):
    #     if m.size() == (1, 1):
    #         no_wd.add(m)
    # yes_wd = all_params - no_wd
    
    # print('*' * 80)
    # print('No Weight Decay')
    # print(no_wd)
    # print('*' * 80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW([{'params': list(no_wd), 'weight_decay': 0}, {'params': list(yes_wd)}], lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingWarmupRestarts(optimizer, 300, max_lr=args.lr, min_lr=min_lr, warmup_steps=args.warmup)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    
    
    summary(model, (3, img_size, img_size))
    
    # print(model)
    
    print()
    print("Beginning training")
    print()
    
    lr = optimizer.param_groups[0]["lr"]
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        args.epochs = final_epoch - (checkpoint['epoch'] + 1)
    
    
    for epoch in tqdm(range(args.epochs)):
        # adjust_learning_rate(optimizer, epoch, args)
        lr = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
        acc1 = validate(val_loader, model, criterion, lr, args, epoch=epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 
            }, 
            os.path.join(save_path, 'checkpoint.pth'))
        logger_dict.print()
        if acc1 > best_acc1:
            print('* Best model upate *')
            best_acc1 = acc1
            
            torch.save({
                    'model_state_dict': model.state_dict()
                }
                           , os.path.join(save_path, 'best.pth'))
 
            
        
        print(f'Best acc1 {best_acc1:.2f}')
        print('*'*80)
        print(Style.RESET_ALL)        
        
        writer.add_scalar("Learning Rate", lr, epoch)
        
        # for i in range(len(model.transformer.scale)):
        #     for idx, scale in enumerate(model.transformer.scale[str(i)]):
                
        #         writer.add_scalar(f"Scale/depth{i}_head{idx}", nn.functional.sigmoid(scale), epoch)
        

        
    print(Fore.RED+'*'*80)
    logger.debug(f'best top-1: {best_acc1:.2f}, final top-1: {acc1:.2f}')
    print('*'*80+Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))


def train(train_loader, model, criterion, optimizer, epoch, scheduler,  args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    mix = ''
    mix_paramter = 0
    
    
    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
                
        # Cutmix only
        if args.cm and not args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                mix = 'cutmix'
                mix_paramter = args.beta        
                slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                output = model(images)
                
                loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                
                if args.lam != 0.:            
                    theta = list(map(CosineSimiliarity, model.theta))
                
                    loss +=  args.lam * sum(theta) 
                
                if args.gam != 0.:            
                    theta = list(map(Identity, model.theta))
                
                    loss +=  args.gam * sum(theta)   
                                     
            else:
                mix = 'none'
                mix_paramter = 0
                output = model(images)
                
                loss = criterion(output, target)
                               
                if args.lam != 0.:  
                    theta = list(map(CosineSimiliarity, model.theta))
                    
                    loss +=  args.lam * sum(theta)
                    
                if args.gam != 0.:            
                    theta = list(map(Identity, model.theta))
                
                    loss +=  args.gam * sum(theta)   
        
        # Mixup only
        elif not args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                mix = 'mixup'
                mix_paramter = args.alpha
                images, y_a, y_b, lam = mixup_data(images, target, args)
                output = model(images)
                
                loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                
                if args.lam != 0.: 
                    theta = list(map(CosineSimiliarity, model.theta))
                    
                    loss +=  args.lam * sum(theta)
                    
                if args.gam != 0.:            
                    theta = list(map(Identity, model.theta))
                
                    loss +=  args.gam * sum(theta)   
                    
            
            else:
                mix = 'none'
                mix_paramter = 0
                output = model(images)
                
                loss =  criterion(output, target)
                 
                if args.lam != 0.:
                    theta = list(map(CosineSimiliarity, model.theta))
                
                    loss +=  args.lam * sum(theta)
                    
                if args.gam != 0.:            
                    theta = list(map(Identity, model.theta))
                
                    loss +=  args.gam * sum(theta)   
        
        # Both Cutmix and Mixup
        elif args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                switching_prob = np.random.rand(1)
                
                # Cutmix
                if switching_prob < 0.5:
                    mix = 'cutmix'
                    mix_paramter = args.beta
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                    images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(images)
                    
                    loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                    
                    if args.lam != 0.:
                        theta = list(map(CosineSimiliarity, model.theta))
                    
                        loss +=  args.lam * sum(theta)   
                            
                    if args.gam != 0.:            
                        theta = list(map(Identity, model.theta))
                        
                        loss +=  args.gam * sum(theta)         
                
                # Mixup
                else:
                    mix = 'mixup'
                    mix_paramter = args.alpha
                    images, y_a, y_b, lam = mixup_data(images, target, args)
                    output = model(images)
                    
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam) 
                    
                    if args.lam != 0.:
                    
                        theta = list(map(CosineSimiliarity, model.theta))
                        
                        loss += args.lam * sum(theta)      
                            
                    if args.gam != 0.:            
                        theta = list(map(Identity, model.theta))
                    
                        loss +=  args.gam * sum(theta)                         
            
            else:
                mix = 'none'
                mix_paramter = 0
                output = model(images)
                
                loss = criterion(output, target) 
          
                if args.lam != 0.:
                
                    theta = list(map(CosineSimiliarity, model.theta))
                    
                    loss += args.lam * sum(theta)
                    
                if args.gam != 0.:            
                    theta = list(map(Identity, model.theta))
                
                    loss +=  args.gam * sum(theta)   
           
        # No Mix
        else:
            mix = 'none'
            mix_paramter = 0
            output = model(images)
                                
            loss = criterion(output, target)
                                
            if args.lam != 0.:
            
                theta = list(map(CosineSimiliarity, model.theta))
                
                loss += args.lam * sum(theta)
            
            if args.gam != 0.:            
                theta = list(map(Identity, model.theta))
            
                loss +=  args.gam * sum(theta)   
    
        acc = accuracy(output, target, (1,))
        acc1 = acc[0]
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            progress_bar(i, len(train_loader),f'[Epoch {epoch+1}/{args.epochs}][T][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.7f}   Mix: {mix} ({mix_paramter})'+' '*10)

        # for i in range(3):
        #     print('=======================')
        #     for j in range(4):
        #         print(model.scale[i][j].item())
    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/train", avg_acc1, epoch)
    
    return lr


def validate(val_loader, model, criterion, lr, args, epoch=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            
            output = model(images)
            loss = criterion(output, target)
            
            if args.lam != 0.:
            
                theta = list(map(CosineSimiliarity, model.theta))
                
                loss += args.lam * sum(theta)
                
            if args.gam != 0.:            
                theta = list(map(Identity, model.theta))
            
                loss +=  args.gam * sum(theta)   
            
                
            
            acc = accuracy(output, target, (1, 5))
            acc1 = acc[0]
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                progress_bar(i, len(val_loader), f'[Epoch {epoch+1}][V][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.6f}')
    print()        

    # total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(Fore.BLUE)
    print('*'*80)
    # logger.debug(f'[Epoch {epoch+1}] \t Top-1 {avg_acc1:6.2f} \t lr {lr:.6f} \t Time: {total_mins:.2f}')
    
    logger_dict.update(keys[2], avg_loss)
    logger_dict.update(keys[3], avg_acc1)
    # for i in range(4):
    #     logger_dict.update(keys[4+i], model.scale[0][i].item())
    # for i in range(4):
    #     logger_dict.update(keys[8+i], model.scale[1][i].item())
    # for i in range(4):
    #     logger_dict.update(keys[12+i], model.scale[2][i].item())
    
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Acc/val", avg_acc1, epoch)
    

    
    return avg_acc1


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer
    
    # random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model_name = args.model + f"-{args.depth}-{args.heads}-{args.channel}-{args.tag}-{args.dataset}-LR[{args.lr}]"

    if args.is_base:
        model_name += "-Base"
    else:
        model_name += "-STT"
 
    if args.is_coord:
        model_name += "-Coord"
    
    if args.is_rpe:
        model_name += "-iRPE"
    
    if args.is_ape:
        model_name += "-APE"
 
    if args.is_LSA:
        model_name += "-LSA"
 
    if args.gam > 0.:
        model_name += f"-Iden[{args.gam}]"
 
    if args.lam > 0.:
        model_name += f"-Sim[{args.lam}]"
 
    if args.scale > 0.:
        model_name += f"-Scale[{args.scale}]"
        
    model_name += f"-STT_head[{args.STT_head}]"
    model_name += f"-STT_depth[{args.STT_depth}]"
    model_name += f"-N_trans[{args.n_trans}]"
    model_name += f"-Pe_dim[{args.pe_dim}]"
    model_name += f"-Merge[{args.merging_size}]"
    model_name += f"-Seed{args.seed}"
    save_path = os.path.join(os.getcwd(), 'save', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard', model_name))
    
    # logger

    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    
    global logger_dict
    global keys
    
    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1', 
            'ParameterScale_1', 'ParameterScale_2', 'ParameterScale_3', 'ParameterScale_4',
            'ParameterScale_5', 'ParameterScale_6', 'ParameterScale_7', 'ParameterScale_8',
            'ParameterScale_9', 'ParameterScale_10', 'ParameterScale_11', 'ParameterScale_12']
    
    main(args)
