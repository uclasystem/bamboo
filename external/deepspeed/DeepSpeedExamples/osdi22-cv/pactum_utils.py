#!/usr/bin/env python3
'''
Project pactum utils for dataloading and argparsing.
'''

import os
import argparse

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import deepspeed


DATASET = 'IMAGENET'


def cifar_trainset(local_rank, dl_path='/tmp/cifar10-data'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to args.local_rank
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def imagenet_trainset(local_rank, dl_path='/home/ubuntu/filepool/datasets/imagenet'):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to local_rank
    # Data loading code
    traindir = os.path.join(dl_path, 'train')
    valdir = os.path.join(dl_path, 'val')

    trainset = torchvision.datasets.ImageFolder(traindir, transform)
    return trainset


def get_trainset(local_rank, dataset='imagenet', dl_path='/home/ubuntu/filepool/datasets/imagenet'):
    if dataset == 'imagenet':
        DATASET = 'IMAGENET'
        return imagenet_trainset(local_rank, dl_path=dl_path)
    elif dataset == 'cifar':
        DATASET = 'CIFAR'
        return cifar_trainset(local_rank)

    print("ERROR when get trainset! Don't support dataset " + dataset + "!")
    exit(-1)


def get_args():
    parser = argparse.ArgumentParser(description=DATASET)
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')

    ## Project pactum: for elasticity
    parser.add_argument('-rl', '--redundancy_level',
                        type=int,
                        default=0)
    parser.add_argument('--eager',
                        action='store_true',
                        default=False)
    parser.add_argument('--debug',
                        action='store_true',
                        default=False)
    parser.add_argument('--mem_log',
                        action='store_true',
                        default=False)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def init_rdzv(args):
    import json
    from torch.distributed.elastic.rendezvous import RendezvousParameters
    from project_pactum.rendezvous.etcd import create_rdzv_handler

    rdzv_backend = 'etcd-v2'
    rdzv_endpoint = os.environ['PROJECT_PACTUM_ENDPOINT']
    run_id = os.environ['PROJECT_PACTUM_RUN_ID']
    min_nodes = int(os.environ['PROJECT_PACTUM_MIN_NODES'])
    max_nodes = int(os.environ['PROJECT_PACTUM_MAX_NODES'])
    rdzv_configs = json.loads(os.environ['PROJECT_PACTUM_RDZV_CONFIGS'])
    rdzv_configs['last_call_timeout'] = 1
    rdzv_parameters = RendezvousParameters(
        backend=rdzv_backend,
        endpoint=rdzv_endpoint,
        run_id=run_id,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        **rdzv_configs,
    )
    rdzv_handler = create_rdzv_handler(rdzv_parameters)
    args.rdzv_handler = rdzv_handler

    return rdzv_handler


def deepspeed_init_distributed(args, rdzv_handler):
    deepspeed.init_distributed(args.backend, rank=int(os.environ['RANK']), world_size=int(
        os.environ['WORLD_SIZE']), store=rdzv_handler.setup_kv_store())
