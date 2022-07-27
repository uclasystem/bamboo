#!/usr/bin/env python3

import os
import argparse

import torch
import torch.distributed as dist

from torchvision.models import AlexNet

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from deepspeed.runtime.pipe.topology import ProcessTopology, PipeDataParallelTopology

# Project pactum utils for dataloading and argparsing
from pactum_utils import *


def train_base(args):
    torch.manual_seed(args.seed)

    net = AlexNet(num_classes=10)

    trainset = get_trainset(args.local_rank, dataset='imagenet')

    engine, _, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    dataloader = RepeatingLoader(dataloader)
    data_iter = iter(dataloader)

    rank = dist.get_rank()
    gas = engine.gradient_accumulation_steps()

    criterion = torch.nn.CrossEntropyLoss()

    total_steps = args.steps * engine.gradient_accumulation_steps()
    step = 0
    for micro_step in range(total_steps):
        batch = next(data_iter)
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)

        outputs = engine(inputs)
        loss = criterion(outputs, labels)
        engine.backward(loss)
        engine.step()

        if micro_step % engine.gradient_accumulation_steps() == 0:
            step += 1
            if rank == 0 and (step % 10 == 0):
                print(f'step: {step:3d} / {args.steps:3d} loss: {loss}')


def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers


def train_pipe(args, part='custom'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    #
    # Build the model
    #
    net = AlexNet(num_classes=10)

    ## Project pactum: customize pipeline topology
    data_parallel_size = int(os.environ['PROJECT_PACTUM_NUM_PIPELINES'])
    pipe_parallel_size = int(os.environ['PROJECT_PACTUM_NUM_STAGES'])
    ## rdzv_handler initialized in `init_rdzv_and_deepspeed()`
    global_decision = args.rdzv_handler.get_global_decision()
    custom_topology = PipeDataParallelTopology(
        pipe_parallel_size, data_parallel_size, global_decision)

    custom_partition = [0, 4, 8, 12, 17, 20, 22]
    net = PipelineModule(layers=join_layers(net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=pipe_parallel_size,
                         topology=custom_topology,
                         partition_method=part,
                         custom_partitions=custom_partition,
                         activation_checkpoint_interval=0)

    trainset = get_trainset(args.local_rank, dataset='imagenet', dl_path='/home/ubuntu/filepool/datasets/imagenette2')

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset,
        ## Project pactum: extra params for redundancy and recovery
        redundancy_level=args.redundancy_level,
        eager_recovery=args.eager)

    for step in range(args.steps):
        loss = engine.train_batch()


if __name__ == '__main__':
    args = get_args()

    ## Project pactum: init rdzv handler
    rdzv_handler = init_rdzv(args)
    ## Project pactum: deepspeed.init_distributed
    deepspeed_init_distributed(args, rdzv_handler)

    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        train_pipe(args)
