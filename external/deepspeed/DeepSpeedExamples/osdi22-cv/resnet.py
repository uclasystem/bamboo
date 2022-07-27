#!/usr/bin/env python3

import os
import argparse

import torch
import torch.distributed as dist

from torchvision.models import resnet152

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from deepspeed.runtime.pipe.topology import ProcessTopology, PipeDataParallelTopology
from colorama import Fore

# Project pactum utils for dataloading and argparsing
from pactum_utils import *

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(1, -1).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_base(args):
    torch.manual_seed(args.seed)

    net = resnet152().cuda()

    trainset = get_trainset(args.local_rank, dataset='imagenet', dl_path='/home/ubuntu/filepool/datasets/imagenette2')

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
    )

    criterion = torch.nn.CrossEntropyLoss()

    for s in range(args.steps):
        for i, (images, target) in enumerate(train_loader):
            print('RUNNING STEP ', i)
            images = images.cuda()
            target = target.cuda()

            outputs = net(images)
            loss = criterion(outputs, target)
            acc1 = accuracy(outputs, target, topk=(1,))

            print("Top-1 Acc: ", acc1)

            loss.backward()

            break
        break


def join_resnet_layers(model):
    layers = [
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        *model.layer1,
        *model.layer2,
        *model.layer3,
        *model.layer4,
        model.avgpool,
        lambda x: torch.flatten(x, 1),
        model.fc,
    ]
    return layers


def train_pipe(args, part='uniform'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    #
    # Build the model
    #
    net = resnet152()

    ## Project pactum: customize pipeline topology
    data_parallel_size = int(os.environ['PROJECT_PACTUM_NUM_PIPELINES'])
    pipe_parallel_size = int(os.environ['PROJECT_PACTUM_NUM_STAGES'])
    global_decision = args.rdzv_handler.get_global_decision()
    print(Fore.RED, f'[ {dist.get_rank()} ] GETTING CUSTOM TOPO', Fore.RESET)
    custom_topology = PipeDataParallelTopology(
        pipe_parallel_size, data_parallel_size, global_decision)

    print(Fore.RED, f'[ {dist.get_rank()} ] CREATING PIPE MODULE!!!', Fore.RESET)
    custom_partition = [0, 22, 31, 40, 49, 52, 57]
    net = PipelineModule(layers=join_resnet_layers(net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=pipe_parallel_size,
                         topology=custom_topology,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    print(Fore.RED, f'[ {dist.get_rank()} ] GETTING TRAINSET!!!', Fore.RESET)
    trainset = get_trainset(args.local_rank, dataset='imagenet', dl_path='/home/ubuntu/filepool/datasets/imagenette2')

    print(Fore.RED, f'[ {dist.get_rank()} ] INIT DEEPSPEED!!!', Fore.RESET)
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset,
        ## Project pactum: extra params for redundancy and recovery
        redundancy_level=args.redundancy_level,
        eager_recovery=args.eager)

    for step in range(args.steps):
        loss = engine.train_batch(debug=args.debug)


if __name__ == '__main__':
    args = get_args()

    ## Project pactum: init rdzv handler
    ## Project pactum: deepspeed.init_distributed

    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        rdzv_handler = init_rdzv(args)
        deepspeed_init_distributed(args, rdzv_handler)
        train_pipe(args)
