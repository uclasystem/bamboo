"""
Reference: - https://raw.githubusercontent.com/weiaicunzai/pytorch-cifar100/master/models/resnet.py
"""

from json import load
import numpy as np
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2 = self._make_layer(block, 64, num_block[0], 1)
        self.conv3 = self._make_layer(block, 128, num_block[1], 2)
        self.conv4 = self._make_layer(block, 256, num_block[2], 2)
        self.conv5 = self._make_layer(block, 512, num_block[3], 2)

        class Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512 * block.expansion, num_classes)

            def forward(self, x):
                output = self.avg_pool(x)
                output = output.view(output.size(0), -1)
                output = self.fc(output)
                return output

        self.final = Head()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        return self.final(output)

    def join_layers(self):
        return [self.conv1] + \
               [i for i in self.conv2] + \
               [i for i in self.conv3] + \
               [i for i in self.conv4] + \
               [i for i in self.conv5] + [self.final]


class DatasetSimple(torch.utils.data.Dataset):
    def __init__(self, size=512):
        self._size = size
        self._inputs = np.random.randn(size, 3, 224, 224)
        self._labels = np.random.randint(0, 100, (size, ))

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return (torch.tensor(self._inputs[idx], dtype=torch.float32),
                self._labels[idx].astype('long'))


def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])


def test():
    from torch.utils.data import DataLoader

    device = 'cuda:0'
    model = resnet50().to(device)

    batch_size = 4
    dataset = DatasetSimple()
    loader = DataLoader(dataset, batch_size=batch_size)

    label = torch.randint(0, 100, (batch_size, ), dtype=torch.long).to(device)
    loss_fn = nn.CrossEntropyLoss()

    for batch in loader:
        x = batch[0].to(device)
        label = batch[1].to(device)
        break

    out = model(x)
    loss = loss_fn(out, label)

    print(f'Input shape: {x.shape}; \n'
          f'Output shape: {out.shape}; \n'
          f'Loss: {loss:.4f}')


def train():
    import deepspeed
    from deepspeed.pipe import PipelineModule
    from deepspeed.runtime.pipe.topology import PipeDataParallelTopology

    def get_args():
        import argparse
        parser = argparse.ArgumentParser(description='FFN')
        parser.add_argument('-s',
                            '--steps',
                            type=int,
                            default=10,
                            help='quit after this many steps')
        parser.add_argument('--curr-step', '-cs',
                            type=int,
                            default=0,
                            help='The step to start on')
        parser.add_argument('--backend',
                            type=str,
                            default='nccl',
                            help='distributed backend')

        ## Elastic stuff
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
        parser.add_argument('--profile_step',
                            type=int,
                            default=-1,
                            help='step to enable pytorch profiler; -1 means disable')
        parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')

        # Model config args
        parser.add_argument('--parts',
                            type=str,
                            default='6,6,6,6,6,7,7,8',
                            help='Specify number of layers for each partition; separated by comma like `1,2,2,3`')
        parser.add_argument('--aci',
                            type=int,
                            default=2,
                            help='Activation checkpoint interval')

        parser = deepspeed.add_config_arguments(parser)
        args = parser.parse_args()
        return args
    args = get_args()
    np.random.seed(args.seed)

    def init_dist(args):
        import json
        import os
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
        deepspeed.init_distributed(
            args.backend,
            rank=int(os.environ['RANK']),
            world_size=int(os.environ['WORLD_SIZE']),
            store=rdzv_handler.setup_kv_store())

        data_parallel_size = int(os.environ['PROJECT_PACTUM_NUM_PIPELINES'])
        pipe_parallel_size = int(os.environ['PROJECT_PACTUM_NUM_STAGES'])
        global_decision = rdzv_handler.get_global_decision()
        custom_topology = PipeDataParallelTopology(pipe_parallel_size, data_parallel_size, global_decision)
        return {'data_parallel_size': data_parallel_size,
                'pipe_parallel_size': pipe_parallel_size,
                'topo': custom_topology}
    dist_config = init_dist(args)

    def gen_parts(args):
        parts = []
        if args.parts:
            parts = [int(p) for p in args.parts.split(',')]
            parts = [0] + [sum(parts[:i]) + p for i, p in enumerate(parts)]
        return parts
    parts = gen_parts(args)

    layers = resnet152().join_layers()
    model = PipelineModule(layers=layers,
                           loss_fn=nn.CrossEntropyLoss(),
                           num_stages=dist_config['pipe_parallel_size'],
                           partition_method='uniform' if len(parts) == 0 else 'custom',
                           custom_partitions=parts,
                           topology=dist_config['topo'],
                           activation_checkpoint_interval=args.aci)

    dataset = DatasetSimple()

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset,
        redundancy_level=args.redundancy_level,
        eager_recovery=args.eager)

    for i in range(engine.global_steps, args.steps):
        if args.profile_step == i:
            from pathlib import Path
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                engine.train_batch(debug=args.debug, mem_log=args.mem_log)

            trace_file = Path.home() / Path('trace.json')
            prof.export_chrome_trace(str(trace_file))
            print(f'[DEBUG Profiler] write to {trace_file}')
        else:
            engine.train_batch(debug=args.debug, mem_log=args.mem_log)


if __name__ == '__main__':
    train()
