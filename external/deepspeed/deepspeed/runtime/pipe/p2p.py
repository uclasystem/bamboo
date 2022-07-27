'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import datetime
import torch
import torch.distributed as dist

from typing import List, Optional

_groups: List[List[Optional[dist.ProcessGroup]]] = []  # [src_stage][recv_stage]
_grid = None
_fallback_groups = None


# initializes adjacent process groups
# run this only after torch.distributed.init_process_group() has been called
def init_process_groups(grid, device):
    global _groups, _grid
    _grid = grid

    assert _grid.pipe_parallel_size > 1, "There is no pipeline parallelism"
    _groups = [[] for _ in range(dist.get_world_size())]  # [src_rank][dst_rank]

    # For each pipeline
    for i in range(grid.get_data_parallel_world_size()):
        # For each stage
        for j in range(grid.get_pipe_parallel_world_size()):
            src_rank = grid.topology().get_rank(data=i, pipe=j)
            src_groups = [None for _ in range(dist.get_world_size())]
            partners = grid.p2p_matrix[src_rank]
            for dst_rank in partners:
                if src_rank in _groups[dst_rank] and \
                        _groups[dst_rank][src_rank] is not None:
                    src_groups[dst_rank] = _groups[dst_rank][src_rank]
                else:
                    new_group = dist.new_group(ranks=[src_rank, dst_rank])
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
            _groups[src_rank] = src_groups

def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def _set_fallback_status():
    recv.fallback = True
    send.fallback = True


def send(tensor, dest_stage):
    global _grid, _groups

    src_stage = _grid.get_stage_id()

    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)

    dist.broadcast(tensor, src_rank, group=group)
    # Mandatory barrier to detect whether send succeeds
    handler = dist.barrier(group, async_op=True)
    handler.wait(datetime.timedelta(seconds=60))

def recv(tensor, src_stage):
    global _grid, _groups

    dest_stage = _grid.get_stage_id()

    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)

    dist.broadcast(tensor, src_rank, group=group)
    # To be consistent with send
    handler = dist.barrier(group, async_op=True)
    handler.wait(datetime.timedelta(seconds=60))

def barrier(stage_id):
    global _groups, _grid
    group_id = _grid.stage_to_global(stage_id=stage_id)
    if (dist.get_rank() >= 0):
        print("Barrier Group ID", group_id)
        print("Barrier Group", _grid.p2p_groups[group_id])
    dist.barrier(group=_groups[group_id])
    if (dist.get_rank() >= 0):
        print("Exiting Barrier ", group_id)


def _get_send_recv_group(src_stage: int, dest_stage: int) -> dist.ProcessGroup:
    '''the group id is always the smaller rank unless its a wrap around'''
    global _groups

    src_rank = _grid.stage_to_global(stage_id=src_stage)
    dest_rank = _grid.stage_to_global(stage_id=dest_stage)

    group = _groups[src_rank][dest_rank]
    if group is None:
        raise Exception(
            f"Group of rank {src_rank} to rank {dest_rank} does not exist")

    return group
