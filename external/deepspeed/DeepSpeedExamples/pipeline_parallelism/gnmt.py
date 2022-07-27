"""
Reference: https://tsdaemon.github.io/2018/07/08/nmt-with-pytorch-encoder-decoder.html
"""
import copy
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_IDX = 0


class RecurrentEncoderLayer(nn.Module):
    def __init__(self, hidden_size=128, dropout=0.1) -> None:
        super().__init__()
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Get random hidden states
        h = (x.transpose(0, 1).sum(0, keepdim=True)).detach()
        out, _ = self.rnn(x, (h, h))
        return self.dropout(out)


class GNMTSimple(nn.Module):
    def __init__(self, num_layers=8, hidden_size=128, dropout=0.1):
        assert num_layers >= 1, "require at least 1 layers"
        super().__init__()
        self.rnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.rnn_layers.append(RecurrentEncoderLayer(hidden_size))

        self.reduce = lambda x: x.sum(-1)

    def forward(self, x):
        for layer in self.rnn_layers:
            x = layer(x)
        return self.reduce(x)

    def join_layers(self):
        return [i for i in self.rnn_layers] + [self.reduce]


def make_model(*args):
    return GNMTSimple(*args)


def test():
    device = 'cuda:0'

    d_model = 32
    n_layer = 4
    model = make_model(d_model, n_layer).to(device)

    batch_size = 4
    seq_len = 12
    x = torch.randn(batch_size, seq_len, d_model).to(device)

    out = model(x)
    print(f'Input shape: {x.shape}; \n'
          f'Output shape: {out.shape}')


def train():
    import deepspeed
    from deepspeed.pipe import PipelineModule
    from deepspeed.runtime.pipe.topology import PipeDataParallelTopology

    def get_args():
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
        parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')

        # Model config args
        parser.add_argument('-N', type=int, default=16)
        parser.add_argument('--d-model', '-dm', type=int, default=1024)
        parser.add_argument('-seq', type=int, default=256)
        parser.add_argument('--parts',
                            type=str,
                            default='',
                            help='Specify number of layers for each partition; separated by comma like `1,2,2,3`')

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
            assert sum(parts) == args.N
            parts[-1] += 2
            parts = [0] + [sum(parts[:i]) + p for i, p in enumerate(parts)]
        return parts
    parts = gen_parts(args)

    layers = make_model(args.N, args.d_model).join_layers()
    model = PipelineModule(layers=layers,
                           loss_fn=nn.MSELoss(),
                           num_stages=dist_config['pipe_parallel_size'],
                           partition_method='uniform' if len(parts) == 0 else 'custom',
                           custom_partitions=parts,
                           topology=dist_config['topo'],
                           activation_checkpoint_interval=0)

    class DatasetSimple(torch.utils.data.Dataset):
        def __init__(self, seq, d_model, size=2000):
            self._size = size
            self._inputs = np.random.randn(size, seq, d_model)
            self._labels = np.random.randn(size, seq)

        def __len__(self):
            return self._size

        def __getitem__(self, idx):
            return (torch.tensor(self._inputs[idx], dtype=torch.float32),
                    self._labels[idx].astype('float32'))
    dataset = DatasetSimple(args.seq, args.d_model)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset,
        redundancy_level=args.redundancy_level,
        eager_recovery=args.eager)

    for _ in range(engine.global_steps, args.steps):
        engine.train_batch(debug=args.debug, mem_log=args.mem_log)


if __name__ == '__main__':
    train()
