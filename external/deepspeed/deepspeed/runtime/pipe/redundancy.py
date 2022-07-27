""" Toolkit to manage redundant weight copies """
import torch.distributed as dist
import torch
from typing import List, Optional


def get_redundant_stage_ids(
        redundant_level, stage_id, num_stages) -> List[int]:
    """ Get stages that stored a redundant copy on `stage_id` """
    if redundant_level == 0:
        return []

    stage_ids = []
    r_stage_id = stage_id
    for _ in range(redundant_level):
        r_stage_id = (r_stage_id + 1) % num_stages
        stage_ids.append(r_stage_id)

    return stage_ids


def get_redundant_user_stage_ids(
        redundant_level, stage_id, num_stages) -> List[int]:
    """ Get stages that stored a redundant copy of `stage_id` """
    stage_ids = []
    for i in range(num_stages):
        redundant_stage_ids = get_redundant_stage_ids(
            redundant_level, i, num_stages)
        if stage_id in redundant_stage_ids:
            stage_ids.append(i)
    return stage_ids


def create_sync_groups(grid, device, eager_recovery=False) -> List[List[Optional[dist.ProcessGroup]]]:
    backend = 'gloo' if not eager_recovery else 'nccl'
    device = 'cpu' if not eager_recovery else device

    groups = [[] for _ in range(dist.get_world_size())]  # [src_rank][dst_rank]

    # For each pipeline
    for i in range(grid.get_data_parallel_world_size()):
        # For each stage
        for j in range(grid.get_pipe_parallel_world_size()):
            src_rank = grid.topology().get_rank(data=i, pipe=j)
            src_groups = [None for _ in range(dist.get_world_size())]
            partners = grid.p2p_matrix[src_rank]
            for dst_rank in partners:
                if src_rank in groups[dst_rank] and \
                        groups[dst_rank][src_rank] is not None:
                    src_groups[dst_rank] = groups[dst_rank][src_rank]
                else:
                    new_group = dist.new_group(ranks=[src_rank, dst_rank],
                                               backend=backend)
                    src_groups[dst_rank] = new_group

                    # init group communicator
                    if src_rank == dist.get_rank():
                        zero = torch.zeros(1, 1, device=device)
                        rand = torch.randn(1, 1, device=device)
                        dist.broadcast(rand, src=src_rank, group=new_group)
                        dist.broadcast(zero, src=dst_rank, group=new_group)
                    elif dst_rank == dist.get_rank():
                        zero = torch.zeros(1, 1, device=device)
                        rand = torch.randn(1, 1, device=device)
                        dist.broadcast(zero, src=src_rank, group=new_group)
                        dist.broadcast(rand, src=dst_rank, group=new_group)
            groups[src_rank] = src_groups
    return groups
