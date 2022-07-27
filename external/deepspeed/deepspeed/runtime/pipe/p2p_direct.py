import torch.distributed as dist
from .topology import PipelineParallelGrid

_grid: PipelineParallelGrid = None


def init_process_groups(grid, device):
    global _grid
    _grid = grid

def send(tensor, dest_stage):
    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    dist.send(tensor, dst=dest_rank)

def recv(buffer, src_stage):
    src_rank = _grid.stage_to_global(stage_id=src_stage)
    dist.recv(buffer, src=src_rank)
