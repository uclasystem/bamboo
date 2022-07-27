""" Simple feed forward network for testing

Example usage:

```bash
MASTER_ADDR=<master ip addr> \
MASTER_PORT=29500 \
WORLD_SIZE=4 \
RANK=<rank> \
LOCAL_RANK=0 \
python pipeline_parallelism/ffn.py \
    --deepspeed_config pipeline_parallelism/ffn.json
```

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe.topology import ProcessTopology, PipeDataParallelTopology

from torch.distributed.elastic.rendezvous import RendezvousParameters
from project_pactum.rendezvous.etcd import create_rdzv_handler

import argparse
import os
import json


class FeedForwardNode(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        x = F.relu(self.w_2(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, layers):
        super().__init__()
        self.layers = layers
        self.ffns = nn.ModuleList([FeedForwardNode(d_model, d_ff) for _ in range(layers)])

    def forward(self, x):
        for i in range(self.layers):
            x = self.ffns[i](x)
        return x

    def join_layers(self):
        return [i for i in self.ffns] + [lambda x: x.sum(-1)]


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, d_model, size=100000):
        self._size = size
        self._inputs = np.random.randn(size, d_model)
        self._labels = np.random.randn(size)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return (
            torch.tensor(self._inputs[idx], dtype=torch.float32),
            self._labels[idx].astype('float32')
        )


def init_rdzv(args) -> ProcessTopology:
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
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    deepspeed.init_distributed(args.backend, rank=rank, world_size=world_size, store=rdzv_handler.setup_kv_store())

    data_parallel_size = int(os.environ['PROJECT_PACTUM_NUM_PIPELINES'])
    pipe_parallel_size = int(os.environ['PROJECT_PACTUM_NUM_STAGES'])
    global_decision = rdzv_handler.get_global_decision()
    return PipeDataParallelTopology(pipe_parallel_size, data_parallel_size, global_decision)

def get_args():
    parser = argparse.ArgumentParser(description='FFN')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='local rank')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=4,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('-r', '--redundancy_level',
                        type=int,
                        default=0)
    parser.add_argument('--eager',
                        action='store_true',
                        default=False)
    parser.add_argument('--debug',
                        action='store_true',
                        default=False)
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def train_base(args):
    torch.manual_seed(args.seed)

    model = FeedForward(d_model=16, d_ff=32, layers=6).cuda()
    dataset = DummyDataset(d_model=16)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2
    )

    loss_fn = nn.MSELoss()

    for i, (inputs, target) in enumerate(train_loader):
        print(f'RUNNING STEP {i} of FFN')
        inputs = inputs.cuda()
        target = target.cuda()

        output = model(inputs)
        break

if __name__ == '__main__':
    args = get_args()

    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        topo = init_rdzv(args)

        # Load model into host memory.
        model = FeedForward(d_model=16, d_ff=32, layers=12)

        # Load partitioned model into device memory
        model = PipelineModule(layers=model.join_layers(),
                            loss_fn=nn.MSELoss(),
                            num_stages=args.pipeline_parallel_size,
                            topology=topo,
                            partition_method='uniform',
                            activation_checkpoint_interval=0)

        # Setup dataset
        np.random.seed(args.seed)
        dataset = DummyDataset(d_model=16)

        engine, _, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            training_data=dataset,
            redundancy_level=args.redundancy_level,
            eager_recovery=args.eager)

        for _ in range(args.steps):
            engine.train_batch(debug=args.debug)
