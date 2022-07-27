# Copyright 2019 The Microsoft DeepSpeed Team

from ast import expr_context
from datetime import timedelta
from glob import glob
from posixpath import join
from re import M, S
import time
import collections
import copy
import datetime
import io
import json
import logging
import numpy as np
import os
import signal
import sys

from colorama import Fore

from types import MethodType
from typing import Dict, List, Optional, Tuple
from requests import delete

import torch
from torch.cuda import default_stream, stream
from torch.distributed.distributed_c10d import _get_global_rank
from torch.autograd import grad
import torch.nn as nn
import torch.distributed as dist

import deepspeed
from deepspeed.utils.logging import logger
from deepspeed.utils.timer import SynchronizedWallClockTimer, ThroughputTimer

from deepspeed.inference.engine import InferenceEngine
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
from ..engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from ..utils import PartitionedTensor, deserialize_object, ensure_directory_exists, serialize_object
from ..dataloader import RepeatingLoader

from .module import LayerSpec, PipelineModule, PipelineError, TiedLayerSpec
from . import p2p_direct as p2p
from . import schedule
from . import redundancy

def ns_to_s(ns):
    return ns / 1000000000

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2

class PreemptionError(Exception):
    def __init__(self):
        pass

class PeerFailureError(Exception):
    def __init__(self):
        pass


def is_even(number):
    return number % 2 == 0

global should_stop
should_stop = False

def sig_handler(signum, frame):
    print('[Engine] Signal handler called with signal', signum)
    global should_stop
    should_stop = True

mem_alloced = 0
mem_cached = 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class PrevStageException(Exception):
    ...


class NextStageException(Exception):
    ...


class AllReduceException(Exception):
    def __init__(self, src, *args: object) -> None:
        super().__init__(*args)
        # Exception source pipeline
        self.src = src


class PipelineEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    def __init__(self, *super_args, redundancy_level=0, sync_save=False,
                 eager_recovery=False, prev_state={}, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        assert self.zero_optimization_stage() < 2, "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        signal.signal(signal.SIGTERM, sig_handler)

        self.join = False

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        assert not self.elasticity_enabled(), "Elasticity is not currently supported" \
            " with pipeline parallelism."

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self._dec(self.stage_id)
        self.next_stage = self._inc(self.stage_id)
        self.stage_ids = [self.stage_id]

        global_decisions = self.rdzv_handler.get_global_decision()
        for info in global_decisions:
            if info.rank == self.global_rank:
                self.coordinates = info.active_coordinates

        #self.gloo_pg = dist.new_group(backend='gloo')
        if not self.is_first(global_decisions):
            self.join = True
            prev_num_stages = self.find_prev_number_stages(global_decisions)
            old_parts = self.module.get_new_partition(prev_num_stages)

            recv_decisions = self.get_recv_decisions(old_parts, self.module.parts, global_decisions)
            send_decisions = self.get_send_decisions(recv_decisions)

            received_state = self.transfer_layers(recv_decisions, send_decisions, prev_state)
            self.module.load_layers(received_state)
            self.load_optimizer_state(received_state, prev_state)
            #self.verify_optimizer()

            #self.compare_model_state()

        self.global_steps = self.rdzv_handler.get_current_step()

        # Set redundancy related config
        if redundancy_level > 1:
            raise NotImplementedError(
                f'R level {redundancy_level} > 1 is not supported now.')
        self.redundancy_level = redundancy_level

        if eager_recovery and redundancy_level == 0:
            raise Exception(
                "Must set redundancy level to enable eager recovery")
        self.eager_recovery = eager_recovery

        self.init_redundancy()

        self.data_iterator = None
        self.batch_fn = None

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(batch_size=self.micro_batch_size *
                                           self.micro_batches,
                                           num_workers=self.dp_world_size,
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        self.is_pipe_partitioned = self.is_model_parallel
        self.is_grad_partitioned = False

        # NOTE: Temporarily disable for development
        # model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        # num_params = sum([p.numel() for p in model_parameters])
        # unique_params = num_params
        # # Subtract tied parameters if we don't own them
        # if self.module.tied_comms:
        #     tied_params = 0
        #     for key, d in self.module.tied_comms.items():
        #         if self.global_rank != min(d['ranks']):
        #             tied_params += sum(p.numel() for p in d['module'].parameters())
        #     unique_params -= tied_params
        # params_tensor = torch.LongTensor(data=[num_params,
        #                                        unique_params]).to(self.device)
        # dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        # params_tensor = params_tensor.tolist()
        # total_params = params_tensor[0]
        # unique_params = params_tensor[1]
        # if self.grid.data_parallel_id == 0:
        #     logger.info(f'RANK={self.global_rank} '
        #                 f'STAGE={self.stage_id} '
        #                 f'LAYERS={self.module._local_stop - self.module._local_start} '
        #                 f'[{self.module._local_start}, {self.module._local_stop}) '
        #                 f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
        #                 f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
        #                 f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        #intialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid, self.device)

        self.init_pipe_buffers()

        self.enable_mem_status = False
        self.recv_weights_work = []

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline[
                'activation_checkpoint_interval']

        self.loss_model = self.module.loss_fn

        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)

        # Initialize schedule constructor
        self._generate_sched = lambda \
            stage_id=self.stage_id: \
            schedule.TrainSchedule(
                micro_batches=self.micro_batches,
                stages=self.num_stages,
                stage_id=stage_id)

        if self.r_stage_ids:
            # TODO(pengzhan): There should be a loop for each r_stage. To
            # simplify implementation, assume there is only one r_stage.
            schedule_cls = schedule.EagerRecoverySchedule if self.eager_recovery else \
                           schedule.LazyRecoverySchedule
            self._generate_sched = lambda \
                schedule_cls=schedule_cls, \
                stage_id=self.next_stage, \
                curr_sched=self._generate_sched: \
                    schedule_cls(
                        curr_sched(),
                        schedule.TrainSchedule(
                            micro_batches=self.micro_batches,
                            stages=self.num_stages,
                            stage_id=stage_id))

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

    def _inc(self, stage_id):
        return (stage_id + 1) % self.num_stages

    def _dec(self, stage_id):
        return (stage_id - 1 + self.num_stages) % self.num_stages

    def init_redundancy(self):
        self.r_stage_ids = redundancy.get_redundant_stage_ids(
            self.redundancy_level, self.stage_id, self.num_stages)
        self.r_user_stage_ids = redundancy.get_redundant_user_stage_ids(
            self.redundancy_level, self.stage_id, self.num_stages)

        # parameter buffer and optimizer state buffer
        self.param_buffers: Dict[int, Dict[str, nn.Tensor]] = \
            collections.defaultdict(dict)
        self.state_buffers: Dict[int, Dict[str, nn.Tensor]] = \
            collections.defaultdict(dict)

        # guide optimizer distinguish param in different stages
        num_param = \
            len([_ for _ in self.module.get_named_param(self.stage_id)])
        self.ps_map[self.stage_id] = [i for i in range(num_param)]
        self.ps_counter += num_param

        # prepare for weight sync and redundant computation
        if self.redundancy_level > 0:
            for stage_id in self.r_stage_ids:
                device = 'cpu'
                if self.eager_recovery:
                    self.module.build_layers(stage_id)
                    device = self.device

                self.module.allocate_param(
                    stage_id, self.param_buffers[stage_id],
                    device)
                super().allocate_state(
                    self.param_buffers[stage_id].values(),
                    self.state_buffers[stage_id])

            self.grid.init_fallback_group(self.redundancy_level)

    def init_pipe_buffers(self):
        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers: Dict[str, List[nn.Tensor]] = \
            collections.defaultdict(list)
        self.pipe_buffers = {
            'data': [],                     # data from batch input
            'label': [],                    # labels from batch input
            f'input_{self.stage_id}': [],   # received activations
            f'output_{self.stage_id}': [],  # computed activations
            # 'output_tensors': [],  # tensor object to preserve backward graph
        }
        # For all stages except first stage, we will store gradient of retired
        # input tensor in this buffer, which can be used to recover if previous
        # node fails.
        if self.stage_id != 0:
            self.pipe_buffers[f'input_grad_{self.stage_id}'] = []
        # For eager recovery, we also need to create buffer for r_stage.
        if self.eager_recovery:
            for stage_id in self.r_stage_ids:
                self.pipe_buffers[f'input_{stage_id}'] = []
                self.pipe_buffers[f'output_{stage_id}'] = []

        # Create mutables
        self.pipe_recv_buf = None
        self.ping_tensor = torch.randn([1]).cuda()
        self.ping_buffer = torch.randn([1]).cuda()
        self.grad_layer = None
        self.meta_buffer = None

        # The last stage will never send actual activation so
        # `first_output_send` will remain True. But if the first node
        # failed, it will start to send actual activations. In this case,
        # we should skip send tensor meta.
        self.first_output_send = True if self.stage_id != self.num_stages - 1 else False
        self.first_gradient_send = True

    def log(self, msg, color=None):
        output = f'[ {self.global_rank:02d}|{self.global_steps:02d} ] {msg}'
        if color is not None:
            print_color = Fore.RED
            if color == 'r':
                print_color = Fore.RED
            elif color == 'b':
                print_color = Fore.RED
            elif color == 'lb':
                print_color = Fore.LIGHTCYAN_EX
            elif color == 'g':
                print_color = Fore.GREEN
            elif color == 'lg':
                print_color = Fore.LIGHTGREEN_EX
            print(print_color + output + Fore.RESET)
        else:
            print(output)

    def trigger_kill(self, dp_id, stage_id, step, fatal=False):
        if self.global_steps == step and \
          self.grid.get_data_parallel_id() == dp_id and \
          self.stage_id == stage_id:
            if fatal:
                self.log('FAILING', color='r')
                sys.exit(13)
            else:
                self.log('PREEMPTION SOON', color='r')
                os.kill(os.getpid(), signal.SIGTERM)

    def should_kill(self, dp_id, stage_id):
        return self.grid.get_data_parallel_id() == dp_id and \
          self.stage_id == stage_id

    def is_first(self, globlal_decisions):
        '''
            Check if this is the first initialization of the cluster
            or if there are existing nodes which have more up to date
            state
        '''
        for info in globlal_decisions:
            if len(info.previous_coordinates) != 0:
                return False

        return True

    def find_prev_number_stages(self, global_decisions):
        num_prev_stages = 0
        for info in global_decisions:
            for dp_id, s_id in info.previous_coordinates:
                num_prev_stages = max(num_prev_stages, s_id)

        return num_prev_stages + 1

    def add_optimizer_state(self, l_id, lyr, state):
        optim_state = state[l_id][1]
        if hasattr(lyr, 'parameters'):
            for j, p in enumerate(lyr.parameters()):
                self.optimizer.state[p] = optim_state[j]

    def load_optimizer_state(self, recvd_state, prev_state={}):
        bucket = self.module.func_buckets[self.stage_id]
        for i, l_id in enumerate(range(self.module.parts[self.stage_id], self.module.parts[self.stage_id + 1])):
            if l_id in prev_state:
                lyr = bucket[i]
                self.add_optimizer_state(l_id, lyr, prev_state)
                continue

            if l_id in recvd_state:
                lyr = bucket[i]
                self.add_optimizer_state(l_id, lyr, recvd_state)
                continue

    def write_model_state(self, shared_storage='/mnt/efs/verify-transfer'):
        for stage, bucket in self.module.func_buckets.items():
            start = self.module.parts[stage]

            for i, layer in enumerate(bucket):
                if not hasattr(layer, 'state_dict'):
                    continue

                all_layer_info = {}
                for k, v in layer.state_dict().items():
                    all_layer_info[k] = { 'param': v }
                    for p in layer.parameters():
                        if p.data_ptr() == v.data_ptr():
                            all_layer_info[k]['state'] = self.optimizer.state[p]

                torch.save(all_layer_info, f'{shared_storage}/layer{i + start}.json')

    def compare_model_state(self, shared_storage='/mnt/efs/verify-transfer'):
        for stage, bucket in self.module.func_buckets.items():
            start = self.module.parts[stage]

            for i, layer in enumerate(bucket):
                if not hasattr(layer, 'state_dict'):
                    continue

                all_layer_info = {}
                for k, v in layer.state_dict().items():
                    all_layer_info[k] = { 'param': v }
                    for p in layer.parameters():
                        if p.data_ptr() == v.data_ptr():
                            all_layer_info[k]['state'] = self.optimizer.state[p]

                previous_state = torch.load(f'{shared_storage}/layer{i + start}.json')

                ## Make sure that previous state and transferred state are EXACTLY the same
                assert len(previous_state) == len(all_layer_info)
                for k in previous_state:
                    assert k in all_layer_info

                    ## Make sure that it has BOTH 'param' and 'state' entries
                    assert len(previous_state[k]) == len(all_layer_info[k])

                    assert torch.equal(previous_state[k]['param'], all_layer_info[k]['param'])

                    try:
                        for state_tensor_name in previous_state[k]['state']:
                            assert torch.equal(previous_state[k]['state'][state_tensor_name], all_layer_info[k]['state'][state_tensor_name])
                    except KeyError as e:
                        print(f'Failed on layer {i + start} with parameter {k}')
                        print(previous_state[k]['state'])
                        print()
                        print(all_layer_info[k]['state'])
                        raise e

    def transfer_layers(self, recv_decisions, send_decisions={}, prev_state={}):
        received_state = {}

        ## Implement sync transfer protocol that I just though of
        my_send_decicions = send_decisions[self.global_rank] if self.global_rank in send_decisions else {}
        my_recv_decisions = recv_decisions[self.global_rank] if self.global_rank in recv_decisions else {}
        for rank in range(self.world_size):
            if rank == self.global_rank:
                for dst_rank, layer_idxs in my_send_decicions.items():
                    if len(layer_idxs) == 0:
                        continue

                    self.send_layers(dst_rank, sorted(layer_idxs), prev_state)
            else:
                if rank in my_recv_decisions:
                    src_rank = rank
                    layer_idxs = my_recv_decisions[rank]
                    received_state.update(self.recv_layers(src_rank, sorted(layer_idxs)))

        return received_state

    def remove_param_optim_state(self, param):
        del self.optimizer.state[param]

        for i in range(len(self.optimizer.param_groups[0]['params'])):
            t = self.optimizer.param_groups[0]['params'][i]
            ## Annoying way to find the parameters in the param groups
            ## Have to compare the underlying data ptr
            if t.data_ptr() == param.data_ptr():
                del self.optimizer.param_groups[0]['params'][i]
                break

    def get_layer_and_optim_state(self, stage_part_start, func_bucket, delete_state):
        bucket_state = {}
        for i, lyr in enumerate(func_bucket):
            if not hasattr(lyr, 'state_dict'):
                bucket_state[stage_part_start + i] = ({}, [])
                continue

            lyr_optim_state = []
            if hasattr(lyr, 'parameters'):
                for p in lyr.parameters():
                    lyr_optim_state.append(self.optimizer.state[p])
                    if delete_state:
                        self.remove_param_optim_state(p)

            bucket_state[stage_part_start + i] = (lyr.state_dict(), lyr_optim_state)

        return bucket_state

    def get_model_state(self, delete_state=True):
        model_state = {}
        for stage_id in self.stage_ids:
            stage_part_start = self.module.parts[stage_id]
            stage_partition = self.module.func_buckets.get(stage_id, None)
            model_state.update(self.get_layer_and_optim_state(stage_part_start, stage_partition, delete_state))

        return model_state

    def get_recv_decisions(self, old_parts, new_parts, global_decisions):
        recv_decisions = {}
        for info in global_decisions:
            if len(info.active_coordinates) == 0:
                continue

            rank = info.rank
            rank_recv_decisions = {}
            my_stage = info.active_coordinates[0][1]
            needed_layers = set(range(new_parts[my_stage], new_parts[my_stage + 1]))

            if len(info.previous_coordinates) != 0:
                prev_stages = [s_id for dp_id, s_id in info.previous_coordinates]
                prev_partition = set(range(old_parts[min(prev_stages)], old_parts[max(prev_stages) + 1]))
                needed_layers.difference_update(prev_partition)

            for other_info in global_decisions:
                other_rank = other_info.rank

                if len(other_info.previous_coordinates) == 0:
                    continue

                other_prev_stages = [s_id for dp_id, s_id in other_info.previous_coordinates]
                part_start = old_parts[min(other_prev_stages)]
                part_end = old_parts[max(other_prev_stages) + 1]
                other_prev_part = set(range(part_start, part_end))

                intersect = needed_layers.intersection(other_prev_part)
                if len(intersect) == 0: continue

                rank_recv_decisions[other_rank] = intersect
                needed_layers.difference_update(intersect)
                if len(needed_layers) == 0:
                    break

            recv_decisions[rank] = rank_recv_decisions

            assert len(needed_layers) == 0

        return recv_decisions

    def get_send_decisions(self, recv_decisions):
        send_decisions = {}
        for recving_rank, recv_info in recv_decisions.items():
            for sending_rank, layers in recv_info.items():
                if sending_rank not in send_decisions:
                    send_decisions[sending_rank] = {}

                send_decisions[sending_rank][recving_rank] = layers

        return send_decisions

    def reset_process_groups(self, store):
        dist.destroy_process_group()
        deepspeed.init_distributed(self.dist_backend, rank=self.global_rank, world_size=self.world_size, store=store)

    def reconfigure_cluster(self, store, global_decisions, recvd_state):
        self.rdzv_handler.write('/rdzv/cluster_status', 'init')
        self.rdzv_handler.write('/rdzv/last_reconfig', self.global_steps)

        my_prev_state = self.get_model_state()
        my_prev_state.update(recvd_state)

        self.reset_process_groups(store)

        custom_proc_topology = PipeDataParallelTopology(self.num_stages, self.num_pipelines, global_decisions)
        model = PipelineModule(layers=self.module._layer_specs,
                                     loss_fn=self.module.loss_fn,
                                     num_stages=self.num_stages,
                                     topology=custom_proc_topology,
                                     partition_method=self.module.partition_method,
                                     activation_checkpoint_interval=self.module.activation_checkpoint_interval)

        ## Re-init pipeline engine for consistency
        self.__init__(
            args=self.init_args,
            model=model,
            training_data=self.training_data,
            mpu=model.mpu(),
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            redundancy_level=self.redundancy_level,
            eager_recovery=self.init_args.eager,
            prev_state=my_prev_state,
            rdzv_handler=self.rdzv_handler,
        )

    def save_shadow_node_state(self, failures):
        transfer_needed = False
        for fail_rank in failures:
            recv_decisions = {}
            send_decisions = {}
            recv_node_index = 0
            if failures[fail_rank] == self.global_steps:
                prev_rdzv_state = self.rdzv_handler.get_previous_state()
                rank_coordinates = self.rdzv_handler.get_rank_coordinates_for_version(prev_rdzv_state, prev_rdzv_state["version"])

                failing_shadow_node = False
                for rank, coordinates in rank_coordinates.items():
                    if rank == fail_rank and len(coordinates) == 2:
                        failing_shadow_node = True
                        break

                if not failing_shadow_node:
                    continue

                ## A layer transfer should be triggered. To be safe just do it all the time
                ## as opposed to only when its an active node
                send_node = int(rank)
                recv_node = -1

                available_nodes = []
                coords_to_send = []
                for rank, coordinates in rank_coordinates.items():
                    ## Create list of nodes that can recv these layers
                    ## Not about to fail and not already failed
                    if str(rank) not in failures and self.global_store.get(str(rank)) == b'0':
                        available_nodes.append(rank)

                    if failures.get(str(rank), -1) == self.global_steps:
                        coords_to_send = coordinates

                recv_node = int(available_nodes[recv_node_index])
                recv_node_index += 1
                existing_stages = []
                for rank, coordinates in rank_coordinates.items():
                    if rank == recv_node:
                        existing_stages = [s_id for dp_id, s_id in coordinates]
                        break

                stages_to_send = [s_id for dp_id, s_id in coords_to_send if s_id not in existing_stages]
                if not stages_to_send:
                    continue

                assert recv_node != -1
                layers_to_transfer = set(range(self.module.parts[min(stages_to_send)], self.module.parts[max(stages_to_send) + 1]))
                rank_recv_decisions = {}
                rank_recv_decisions[send_node] = layers_to_transfer
                rank_send_decisions = {}
                rank_send_decisions[recv_node] = layers_to_transfer
                recv_decisions[recv_node] = rank_recv_decisions
                send_decisions[send_node] = rank_send_decisions

                transfer_needed = True

                new_coordinates = [[0, s_id] for s_id in stages_to_send]
                updated_coordinates = rank_coordinates[str(recv_node)]
                updated_coordinates.extend(new_coordinates)

                self.rdzv_handler.update_coordinates_for_version(prev_rdzv_state["version"], recv_node, updated_coordinates)

        recvd_state = {}
        if transfer_needed:
            state_to_transfer = self.get_model_state(delete_state=False)
            recvd_state = self.transfer_layers(recv_decisions, send_decisions, state_to_transfer)

        return recvd_state

    def check_preemptions(self, failures):
        ## Check all ranks that you will have to communicate with in an iteration
        ## Your all-reduce group + previous and next stage in the pipeline
        prev_stage_rank = self.grid._topo.get_rank(data=self.grid.get_data_parallel_id(), pipe=self.prev_stage) if self.prev_stage >= 0 else -1
        next_stage_rank = self.grid._topo.get_rank(data=self.grid.get_data_parallel_id(), pipe=self.next_stage) if self.next_stage < self.num_stages else -1
        ranks_to_check = self.grid.current_dp_group + [prev_stage_rank, next_stage_rank]
        for rank in ranks_to_check:
            rank_key = str(rank)

            if rank_key not in failures:
                continue

            if self.global_steps != failures[rank_key]:
                continue

            if rank == self.global_rank:
                self.global_store.set(str(self.global_rank), '1')
                sys.exit(13)

            ## NextStageException
            if self.next_stage < self.num_stages and rank == self.grid._topo.get_rank(data=self.grid.get_data_parallel_id(), pipe=self.next_stage):
                self.log('Next node is going to fail. Using fallback schedule', color='r')
                if self.next_stage not in self.param_buffers or \
                    self.next_stage not in self.state_buffers:
                    raise RuntimeError(
                        "Doesn't have param or state to recover"
                    )

                # Map coordinate of next node to the rank of this node.
                next_rank = self.grid.stage_to_global(self.next_stage)
                next_coord = self.grid.topology().get_coord(next_rank)
                self.grid.topology().modify_mapping(rank=self.global_rank, **next_coord._asdict())
                failed_step = 0

                # Re-generate schedule
                self._generate_sched = lambda \
                    stage_id=self.next_stage, \
                    curr_sched=self._generate_sched, \
                    failed_step=failed_step, \
                    curr_step=0: \
                    schedule.NextStageFailoverSchedule(
                        curr_sched(),
                        schedule.TrainSchedule(
                            micro_batches=self.micro_batches,
                            stages=self.num_stages,
                            stage_id=stage_id),
                        failed_step=failed_step,
                        curr_step=curr_step)

                # Update module and funcs.
                if not self.eager_recovery:
                    self.module.build_layers(
                        self.next_stage, self.param_buffers[self.next_stage])

                # Update optimizer state
                super().update_state(
                    [p for p in self.module.parameters()
                        if p.requires_grad],
                    self.state_buffers[self.next_stage])

                # Reset buffer
                self.param_buffers.pop(self.next_stage)
                self.state_buffers.pop(self.next_stage)

                # Update current stage information
                self.stage_ids.append(self.next_stage)
                self.r_stage_ids = []

                # Setup pipe buffer for next stage.
                if not self.eager_recovery:
                    self.pipe_buffers[f'input_{self.next_stage}'] = []
                    self.pipe_buffers[f'output_{self.next_stage}'] = []

                # Update global decision
                self.coordinates.append([self.grid.get_data_parallel_id(), self.next_stage])
                self.rdzv_handler.update_coordinates(self.global_rank, self.coordinates)

                # Update neighboring stage information
                self.next_stage = self._inc(self.next_stage)

                # FIXME(pengzhan): There are several cases I have not
                # implemented yet. For a 4 node pipeline (0,1,2,3) with
                # 4 model stages (A,B,C,D):
                # 1. Node 1 fails at iteration n, and Node 0 takes its work.
                # Then Node 2 fails at iteration n+1. Recovery requires Node 2
                # to send weight (stage C) to Node 0.
                # 2. Node 2 fails at iteration n, and Node 1 takes its work.
                # Then Node 1 fails at iteration n+1. Recovery requires Node 1
                # to send weights (stage B and C) to Node 0.

            ## PrevStageException
            elif self.prev_stage >= 0 and rank == self.grid._topo.get_rank(data=self.grid.get_data_parallel_id(), pipe=self.prev_stage):
                self.log('Previous node is going to fail. Using fallback schedule', color='r')
                # Map coordinate of previous node to the rank of the node
                # in front of previous node.
                prev_rank = self.grid.stage_to_global(self.prev_stage)
                prev_coord = self.grid.topology().get_coord(prev_rank)
                fallback_rank = self.grid.stage_to_global(self._dec(self.prev_stage))
                self.grid.topology().modify_mapping(rank=fallback_rank, **prev_coord._asdict())

                failed_step = 0
                self._generate_sched = lambda \
                    stage_id=self.prev_stage, \
                    curr_sched=self._generate_sched, \
                    failed_step=failed_step, \
                    curr_step=0:             \
                    schedule.PrevStageFailoverSchedule(
                        schedule.TrainSchedule(
                            micro_batches=self.micro_batches,
                            stages=self.num_stages,
                            stage_id=stage_id),
                        curr_sched(),
                        failed_step=failed_step,
                        curr_step=curr_step)

            ## AllReduceException
            else:
                self.log('Node in my AllReduce group is going to fail. Using fallback group', color='r')
                # FIXME(pengzhan): Support recursive recovery. Here we assume
                # the topology has not been changed, so we can get corrent rank
                # exceptional node and its previous node. Consider what will
                # happen if we have changed the topology.
                src = self.grid._topo.get_coord(rank=rank).data
                coord = self.grid.topology().get_coord(self.global_rank)
                except_coord = coord._replace(data=src)
                failed_rank = self.grid.topology().get_rank(**except_coord._asdict())
                self.grid.current_dp_group = self.grid.dp_fallback_groups[failed_rank]
                #if except_coord.pipe == 0:
                #    raise NotImplementedError(
                #        "First stage exception can not be handled now")

                # Use failover schedule, which basically skip steps before
                # failed step.
                self._generate_sched = lambda \
                    curr_sched=self._generate_sched, \
                    curr_step=0,                    \
                    failed_step=0:                  \
                    schedule.AllReduceFailoverSchedule(
                        curr_sched(),
                        [[] for _ in curr_sched()],
                        failed_step=failed_step)

                # Map coordinate of exceptional node to the rank of the node
                # in front of exceptional node in its pipeline.
                prev_coord = except_coord._replace(pipe=self._dec(except_coord.pipe))
                prev_rank = self.grid.topology().get_rank(**prev_coord._asdict())
                self.grid.topology().modify_mapping(prev_rank, **except_coord._asdict())

                # Change data-parallel group
                self.grid.set_fallback_group(src)

    def _build_data_iter(self, dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.dp_world_size,
            rank=self.mpu.get_data_parallel_rank(),
            shuffle=False)
        # Build a loader and make it repeating.
        pipe_dataloader = self.deepspeed_io(dataset, data_sampler=sampler)
        pipe_dataloader = RepeatingLoader(pipe_dataloader)
        self.set_dataloader(pipe_dataloader)

    def send_layers(self, dst_rank, layer_idxs, state):
        """ Move a number of layers from this rank to dst rank

        Args:
            dst_rank (int): the destination rank that is receiving layers
            layer_idxs (list): The global ids of the layers
            state (dict): Dict of { layer_idx: }
        """
        print(Fore.LIGHTYELLOW_EX, f"TRANSFERRING {', '.join([str(idx) for idx in layer_idxs])} TO", dst_rank, Fore.RESET)
        layer_bucket = []
        optim_bucket = []
        for idx in layer_idxs:
            layer_state = state[idx][0]
            optim_state = state[idx][1]

            #print('LAYER STATE {}'.format(layer_state))
            for param_tensor in layer_state.values():
                #self.log(f'PARAM TENSOR {param_tensor}')
                layer_bucket.append(param_tensor)

            for optim_dict in optim_state:
                for tensor_value in optim_dict.values():
                    optim_bucket.append(tensor_value)

        layer_tensor = self.flatten(layer_bucket).cuda()
        optim_tensor = self.flatten(optim_bucket).cuda()

        self.log(f'SIZE OF LAYER TENSOR {layer_tensor.size()}', color='lg')
        self.log(f'SIZE OF OPTIM TENSOR {optim_tensor.size()}', color='lg')
        #group = None if layer_tensor.is_cuda else self.gloo_pg

        ## Send the layers and optimizer state
        dist.send(layer_tensor, dst=dst_rank) #, group=group)
        dist.send(optim_tensor, dst=dst_rank) #, group=group)

    def recv_layers(self, src_rank, layer_idxs):
        """ Receive a set of layers from rank src
        Args:
            layer_idxs (list): The global ids of the layers to move
            src_rank (int): The source rank that is sending the layers
        """
        ## JOHN: Start with a simple implementation but hope to eventually use the
        ## same bucketing technique used in the all-reduce to speed up
        layer_bucket = []
        optim_state_bucket = []

        print(Fore.LIGHTYELLOW_EX, f"RECEIVING {', '.join([str(idx) for idx in layer_idxs])} FROM", src_rank, Fore.RESET)
        layer_state_dicts = []
        for idx in layer_idxs:
            layer = self.module._layer_specs[idx]

            if not hasattr(layer, 'parameters'):
                layer_state_dicts.append({})
                continue
            layer_state_dicts.append(layer.state_dict())

            for p in layer.parameters():
                layer_bucket.append(torch.ones_like(p))

                ## Hardcoded for the FusedAdam optimizer which has two optim state
                ## tensors for every parameter
                for _ in range(2):
                    optim_state_bucket.append(torch.ones_like(p))

        layer_tensor = self.flatten(layer_bucket).cuda()
        optim_tensor = self.flatten(optim_state_bucket).cuda()

        self.log(f'SIZE OF LAYER TENSOR {layer_tensor.size()}', color='lg')
        self.log(f'SIZE OF OPTIM TENSOR {optim_tensor.size()}', color='lg')
        #group = None if layer_tensor.is_cuda else self.gloo_pg

        dist.recv(layer_tensor, src=src_rank) #, group=group)
        dist.recv(optim_tensor, src=src_rank) #, group=group)

        recvd_state = {}

        index = 0
        received_lsds = self.unflatten(layer_tensor, layer_bucket)
        for i in range(len(layer_state_dicts)):
            sd = layer_state_dicts[i]
            if len(sd) == 0:
                continue

            for k in sd:
                sd[k] = received_lsds[index]
                index += 1

            recvd_state[layer_idxs[i]] = [sd, []]

        received_optim_tensors = self.unflatten(optim_tensor, optim_state_bucket)
        index = 0
        state_keys = ['exp_avg', 'exp_avg_sq']
        assert len(received_optim_tensors) % len(state_keys) == 0
        for i in range(len(layer_state_dicts)):
            sd = layer_state_dicts[i]
            if len(sd) == 0:
                continue

            optim_state_dicts_list = []
            for v in sd.values():
                optim_state_dict = {}
                for k in state_keys:
                    assert v.size() == received_optim_tensors[index].size()
                    optim_state_dict[k] = received_optim_tensors[index]
                    index += 1

                optim_state_dicts_list.append(optim_state_dict)

            recvd_state[layer_idxs[i]][1] = optim_state_dicts_list

        return recvd_state

    def get_optimizer_state(self, layers):
        optim_state_dicts = []
        for l in layers:
            if hasattr(l, 'parameters'):
                for p in l.parameters():
                    optim_state_dicts.append(self.optimizer.state[p])
            else:
                optim_state_dicts.append({})

        return serialize_object(optim_state_dicts, to_cuda=True)

    def remove_optimizer_state(self, layers):
        for l in layers:
            for p in l.parameters():
                del self.optimizer.state[p]

                for i in range(len(self.optimizer.param_groups[0]['params'])):
                    t = self.optimizer.param_groups[0]['params'][i]
                    ## Annoying way to find the parameters in the param groups
                    ## Have to compare the underlying data ptr
                    if t.data_ptr() == p.data_ptr():
                        del self.optimizer.param_groups[0]['params'][i]
                        break

    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()
        self.module.allreduce_tied_weight_gradients()

    def _exec_reduce_grads(self, stage_id):
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            self.allreduce_gradients(stage_id, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    def _clean_pipe_buffers(self):
        for key in self.pipe_buffers:
            self.pipe_buffers[key] = []
        self.num_pipe_buffers = 0

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        for key in self.pipe_buffers:
            if len(self.pipe_buffers[key]) >= num_buffers:
                continue
            num_added = num_buffers - len(self.pipe_buffers[key])
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = max(self.num_pipe_buffers, num_buffers)

    def train_batch(self, data_iter=None, debug=False, mem_status=False,
                    mem_log=False):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.rdzv_handler.write('/rdzv/cluster_status', 'train')
        self.log(f'STARTING BATCH {self.global_steps} with coordinates {self.coordinates}')

        global should_stop
        if should_stop:
            self.log('------- EXITING', color='r')
            failures = json.loads(self.global_store.get('failures'))
            already_deleted = []
            for rank, step in failures.items():
                if step < self.global_steps:
                    already_deleted.append(rank)

            for rank in already_deleted:
                del failures[rank]

            failures[self.global_rank] = self.global_steps + 1
            self.global_store.set('failures', json.dumps(failures))
            should_stop = False

        failures = json.loads(self.global_store.get('failures'))
        self.log(f'FAILURES {failures}', color='r')

        start_step = time.time()

        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        self.enable_mem_status = mem_status

        if data_iter:
            self.set_dataiterator(data_iter)

        self.module.train()
        self.total_loss = None
        self._compute_loss = True
        if not self.join and self.rdzv_handler.should_reconfigure(self.global_steps, failures):
            ## If a shadow node is going to fail make sure we get its state before it dies
            ## TODO: Make sure this only happens when the state is not available in another
            ##      pipeline
            recvd_state = self.save_shadow_node_state(failures)

            if failures.get(str(self.global_rank), -1) == self.global_steps:
                print("We are reconfiguring and I will die soon anyway. I'm leaving")
                self.global_store.set(str(self.global_rank), '1')
                sys.exit(13)

            self.log(f'Triggered a reconfiguration on global step {self.global_steps}')
            store, rank, world_size, num_pipelines, num_stages, global_decision = self.rdzv_handler.next_rendezvous(self.global_rank)
            self.log(f'Rendezvous complete! rank: {rank}, world_size: {world_size}, num_pipelines: {num_pipelines}, num_stages: {num_stages}')
            self.log(f'Global decision, {global_decision}')
            self.global_rank = rank
            self.world_size = world_size
            self.global_store = store
            self.num_pipelines = num_pipelines
            self.num_stages = num_stages

            for info in global_decision:
                if info.rank == rank:
                    if len(info.active_coordinates) == 0:
                        self.log(f"-------- I'M LEAVING!!", color=True)
                        sys.exit(125)

            Spec = collections.namedtuple('WorkerSpec', ['role', 'local_world_size'])
            spec = Spec('default', 1)
            self.rdzv_handler.assign_worker_ranks(store, rank, world_size, spec, num_pipelines, num_stages, global_decision)
            if self.global_rank == 0:
                self.rdzv_handler.set_master_addr_port(store)
            master_addr, master_port = self.rdzv_handler.get_master_addr_port(store)

            self.reconfigure_cluster(store, global_decision, recvd_state)
            failures = {}

        self.check_preemptions(failures)

        if self.join:
            self.join = False

        # Do the work
        self.timers('train_batch').start()

        # First trail
        sched = self._generate_sched()
        schedule_status: Optional[Tuple[int, Exception]] = \
            self._exec_schedule(sched, debug=debug)

        if schedule_status is None:
            if debug:
                print('[DEBUG Pipeline] Finish one iteration')
        else:
            if debug:
                print(f'[DEBUG Pipeline] Failed at step {schedule_status[0]} '
                      f'due to {schedule_status[1]}')

            self.rdzv_handler.write('/rdzv/last_reconfig', self.global_steps)

            failed_step = 0
            if type(schedule_status[1]) == NextStageException:
                if self.next_stage not in self.param_buffers or self.next_stage not in self.state_buffers:
                    raise RuntimeError("Doesn't have param or state to recover")

                # Map coordinate of next node to the rank of this node.
                next_rank = self.grid.stage_to_global(self.next_stage)
                next_coord = self.grid.topology().get_coord(next_rank)
                self.grid.topology().modify_mapping(rank=self.global_rank, **next_coord._asdict())

                # Coordinate failed step
                self.fail_lock.acquire(blocking=True, lock_ttl=10)
                fail_step = json.loads(self.global_store.get(f'fail-step-{next_rank}'))
                fail_step.append(schedule_status[0])
                self.log(f'FAIL STEP {next_rank} = {fail_step}')
                self.global_store.set(f'fail-step-{next_rank}', json.dumps(fail_step))
                self.fail_lock.release()

                if self.next_stage != self.num_stages - 1:
                    while len(fail_step) < 2:
                        self.fail_lock.acquire(blocking=True, lock_ttl=10)
                        fail_step = json.loads(self.global_store.get(f'fail-step-{next_rank}'))
                        self.fail_lock.release()

                failed_step = min(fail_step)
                self.log(f'FINAL VERSION OF FAIL STEP = {fail_step}. Got FAILED_STEP = {failed_step}')

                # Re-generate schedule
                self._generate_sched = lambda \
                    stage_id=self.next_stage, \
                    curr_sched=self._generate_sched, \
                    failed_step=failed_step, \
                    curr_step=schedule_status[0]: \
                    schedule.NextStageFailoverSchedule(
                        curr_sched(),
                        schedule.TrainSchedule(
                            micro_batches=self.micro_batches,
                            stages=self.num_stages,
                            stage_id=stage_id),
                        failed_step=failed_step,
                        curr_step=curr_step)

                # Update module and funcs.
                if not self.eager_recovery:
                    self.module.build_layers(self.next_stage, self.param_buffers[self.next_stage])

                # Update optimizer state
                super().update_state(
                    [p for p in self.module.parameters() if p.requires_grad],
                    self.state_buffers[self.next_stage])

                # Reset buffer
                self.param_buffers.pop(self.next_stage)
                self.state_buffers.pop(self.next_stage)

                # Update current stage information
                self.stage_ids.append(self.next_stage)
                self.r_stage_ids = []

                # Setup pipe buffer for next stage.
                if not self.eager_recovery:
                    self.pipe_buffers[f'input_{self.next_stage}'] = []
                    self.pipe_buffers[f'output_{self.next_stage}'] = []

                # Update neighboring stage information
                self.next_stage = self._inc(self.next_stage)

                # FIXME(pengzhan): There are several cases I have not
                # implemented yet. For a 4 node pipeline (0,1,2,3) with
                # 4 model stages (A,B,C,D):
                # 1. Node 1 fails at iteration n, and Node 0 takes its work.
                # Then Node 2 fails at iteration n+1. Recovery requires Node 2
                # to send weight (stage C) to Node 0.
                # 2. Node 2 fails at iteration n, and Node 1 takes its work.
                # Then Node 1 fails at iteration n+1. Recovery requires Node 1
                # to send weights (stage B and C) to Node 0.

            elif type(schedule_status[1]) == PrevStageException:
                # Map coordinate of previous node to the rank of the node
                # in front of previous node.
                prev_rank = self.grid.stage_to_global(self.prev_stage)
                prev_coord = self.grid.topology().get_coord(prev_rank)
                fallback_rank = self.grid.stage_to_global(self._dec(self.prev_stage))
                self.grid.topology().modify_mapping(rank=fallback_rank, **prev_coord._asdict())

                # Coordinate failed step
                self.fail_lock.acquire(blocking=True, lock_ttl=10)
                fail_step = json.loads(self.global_store.get(f'fail-step-{prev_rank}'))
                fail_step.append(schedule_status[0])
                self.log(f'FAIL STEP {prev_rank} = {fail_step}')
                self.global_store.set(f'fail-step-{prev_rank}', json.dumps(fail_step))
                self.fail_lock.release()

                if self.prev_stage != 0:
                    while len(fail_step) < 2:
                        self.fail_lock.acquire(blocking=True, lock_ttl=10)
                        fail_step = json.loads(self.global_store.get(f'fail-step-{prev_rank}'))
                        self.fail_lock.release()

                failed_step = min(fail_step)
                self.log(f'FINAL VERSION OF FAIL STEP = {fail_step}. Got FAILED_STEP = {failed_step}')

                self._generate_sched = lambda \
                    stage_id=self.prev_stage, \
                    curr_sched=self._generate_sched, \
                    failed_step=failed_step, \
                    curr_step=schedule_status[0]: \
                    schedule.PrevStageFailoverSchedule(
                        schedule.TrainSchedule(
                            micro_batches=self.micro_batches,
                            stages=self.num_stages,
                            stage_id=stage_id),
                        curr_sched(),
                        failed_step=failed_step,
                        curr_step=curr_step)

            elif type(schedule_status[1]) == AllReduceException:
                # FIXME(pengzhan): Support recursive recovery. Here we assume
                # the topology has not been changed, so we can get corrent rank
                # exceptional node and its previous node. Consider what will
                # happen if we have changed the topology.
                src = schedule_status[1].src
                coord = self.grid.topology().get_coord(self.global_rank)
                except_coord = coord._replace(data=src)
                failed_rank = self.grid.topology().get_rank(**except_coord._asdict())
                self.grid.current_dp_group = self.grid.dp_fallback_groups[failed_rank]
                if except_coord.pipe == 0:
                    raise NotImplementedError(
                        "First stage exception can not be handled now")

                # Use failover schedule, which basically skip steps before
                # failed step.
                self._generate_sched = lambda \
                    curr_sched=self._generate_sched, \
                    curr_step=0,                    \
                    failed_step=schedule_status[0]: \
                    schedule.AllReduceFailoverSchedule(
                        curr_sched(),
                        [[] for _ in curr_sched()],
                        failed_step=failed_step)

                # Map coordinate of exceptional node to the rank of the node
                # in front of exceptional node in its pipeline.
                prev_coord = except_coord._replace(pipe=except_coord.pipe-1)
                prev_rank = self.grid.topology().get_rank(**prev_coord._asdict())
                self.grid.topology().modify_mapping(prev_rank, **except_coord._asdict())

                # Change data-parallel group
                self.grid.set_fallback_group(src)

            else:  # Unknown exception
                raise Exception(schedule_status[1])

            # Second trail
            sched = self._generate_sched()
            schedule_status = self._exec_schedule(sched, debug=debug)

            if schedule_status is None:
                if debug:
                    print('[DEBUG Pipeline] Finish one iteration')

                self._generate_sched = lambda \
                    sched=self._generate_sched: \
                    sched(failed_step=0, curr_step=0)
            else:
                raise Exception(schedule_status[1])

        if self.recv_weights_work:
            for work in self.recv_weights_work:
                work.wait()
        self.recv_weights_work = []

        # TODO(pengzhan): iterative trail
        # completed = False
        # while not completed:
        #     sched = self._generate_sched()
        #     schedule_status: Optional[Tuple[int, Exception]] = \
        #         self._exec_schedule(sched, debug=debug)
        #     if schedule_status is None:
        #         ...
        #     else:
        #         ...

        self.agg_train_loss = self._aggregate_total_loss()

        self.timers('train_batch').stop()

        elapsed = self.timers('train_batch').elapsed(reset=True)
        iter_time = elapsed / self.steps_per_print()
        tput = self.train_batch_size() / iter_time
        if self.global_steps % self.steps_per_print() == 0:
            msg = f'steps: {self.global_steps} ' \
                  f'loss: {self.agg_train_loss:0.4f} ' \
                  f'iter time (s): {iter_time:0.3f} ' \
                  f'samples/sec: {tput:0.3f}'
            if mem_log:
                torch.cuda.synchronize()
                peak_alloc = torch.cuda.max_memory_allocated()
                peak_reserve = torch.cuda.max_memory_reserved()
                curr_alloc = torch.cuda.memory_allocated()
                curr_reserve = torch.cuda.memory_reserved()
                msg += f' peak alloc (MB): {peak_alloc / (1024**2):.3f} ' \
                       f'peak reverse (MB): {peak_reserve / (1024**2):.3f} ' \
                       f'curr alloc (MB): {curr_alloc / (1024**2):.3f} ' \
                       f'peak reverse (MB): {curr_reserve / (1024**2):.3f} '
            logger.info(msg)
            logger.handlers[0].flush()

        # Tensorboard
        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/train_loss',
                                        self.agg_train_loss.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                if self.global_steps % self.steps_per_print() == 0:
                    self.summary_writer.flush()

        if self.wall_clock_breakdown(
        ) and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                'pipe_send_output',
                'pipe_send_grad',
                'pipe_recv_input',
                'pipe_recv_grad'
            ])

        self._clean_pipe_buffers()

        step_end = time.time()
        self.log(f'FINISHING BATCH {self.global_steps} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} took {step_end - start_step} s')

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss

    def eval_batch(self, data_iter, compute_loss=True, reduce_output='avg'):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """

        self.module.eval()

        eval_output = None

        self._compute_loss = compute_loss

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)
        with torch.no_grad():
            self._exec_schedule(sched)

        if self.is_last_stage():
            eval_output = self._reduce_outputs(self.fwd_outputs, reduce=reduce_output)

        if compute_loss:
            eval_output = self._bcast_pipe_scalar(eval_output)

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/eval_loss',
                                        eval_output.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                self.summary_writer.flush()

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()

        return eval_output

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _reduce_outputs(self, outputs, reduce='avg', reduce_dp=True):
        if reduce is None:
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = self._scale_loss_by_gas(reduced)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx],
                                        group=self.mpu.get_data_parallel_group())
                        reduced[idx] /= self.dp_world_size

            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')

    def _bcast_pipe_scalar(self, data, src_rank=None, dtype=torch.float32):
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach()
        else:
            result = torch.Tensor([0.]).type(dtype).to(self.device)

        dist.broadcast(tensor=result,
                       src=src_rank,
                       group=self.mpu.get_pipe_parallel_group())

        return result

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            loss = self._scale_loss_by_gas(self.total_loss)
            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()
            #print(f'RANK={self.global_rank} bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)

            # NOTE: Temporarily disable for development
            # if self.is_data_parallel:
            #     dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
            #     agg_loss /= self.dp_world_size

            # assert self.global_rank in self.grid.pp_group
            # losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            # dist.broadcast(tensor=losses,
            #                src=self.global_rank,
            #                group=self.mpu.get_pipe_parallel_group())

        else:
            # NOTE: Temporarily disable for development
            # Get loss from last stage
            # src_rank = self.grid.stage_to_global(self.num_stages - 1)
            # assert src_rank in self.grid.pp_group
            # losses = torch.Tensor([0., 0.]).to(self.device)
            # dist.broadcast(tensor=losses,
            #                src=src_rank,
            #                group=self.grid.get_pipe_parallel_group())
            # self.dp_group_loss = losses[0].clone().detach()
            # agg_loss = losses[1].clone().detach()
            agg_loss = 0

        return agg_loss

    def set_dataloader(self, loader):
        """"""
        self.training_dataloader = loader
        self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        self.training_dataloader = None
        self.data_iterator = iterator

    def set_batch_fn(self, fn):
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary

    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.data_iterator is not None:
            batch = next(self.data_iterator)

        # Any post-processing, like broadcasting across a slice-parallel group.
        if self.batch_fn:
            batch = self.batch_fn(batch)

        return batch

    def _exec_forward_pass(self, buffer_id, stage_id):
        if self.pipe_buffers[f'output_{stage_id}'][buffer_id] is not None:
            return

        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        input_key = 'data' if stage_id == 0 else f'input_{stage_id}'
        if isinstance(self.pipe_buffers[input_key][buffer_id], tuple):
            inputs = tuple(
                t.clone() for t in self.pipe_buffers[input_key][buffer_id])
        else:
            inputs = self.pipe_buffers[input_key][buffer_id].clone()

        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        self._zero_grads(inputs)

        outputs = super().forward(inputs, stage_id=stage_id)

        if stage_id != self.num_stages - 1:
            self.pipe_buffers[f'output_{stage_id}'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if stage_id == self.num_stages - 1:
            if self._compute_loss and self.loss_model is not None:
                labels = self.pipe_buffers['label'][buffer_id]
                self.loss = self.loss_model(outputs, labels)
            else:
                # Some models just return loss from forward()
                self.loss = outputs

            if isinstance(self.loss, torch.Tensor):
                self.fwd_outputs.append(self.loss.detach())

                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            else:
                self.fwd_outputs.append([l.detach() for l in self.loss])

                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
                    self.total_loss[idx] += l.detach()

        # Free data and label
        if stage_id == 0:
            self.pipe_buffers['data'][buffer_id] = None
        if stage_id == self.num_stages - 1:
            self.pipe_buffers['label'][buffer_id] = None

        self.mem_status('AFTER FWD')

    def _exec_backward_pass(self, buffer_id, stage_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if stage_id == self.num_stages - 1:
            super().backward(self.loss)
            self.mem_status('AFTER BWD')
            return

        outputs = self.pipe_buffers[f'output_{stage_id}'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        grad_tensors = self.grad_layer

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))

        # Retire output tensor. If we need to cover failure of next stage, we
        # will save the output on cpu memory. For eager, we have already used
        # this activation, as redundant forward always happens before
        # backward pass, so we can directly discard it.
        if not self.eager_recovery and self._inc(stage_id) in self.r_stage_ids:
            self.pipe_buffers[f'output_{stage_id}'][buffer_id] = \
                outputs.clone().detach().to('cpu', non_blocking=True)
        else:
            self.pipe_buffers[f'output_{stage_id}'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')

    def _exec_load_micro_batch(self, buffer_id):
        # FIXME: Make it consistent with eager mode
        if self.pipe_buffers['label'][buffer_id] is not None or self.pipe_buffers['data'][buffer_id] is not None:
            return

        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        batch = self._next_batch()

        if 0 in self.stage_ids or 0 in self.r_stage_ids:
            loaded = None
            if torch.is_tensor(batch[0]):
                loaded = batch[0].clone().to(self.device).detach()
                loaded.requires_grad = loaded.is_floating_point()
            else:
                assert isinstance(batch[0], tuple)
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)

            self.pipe_buffers['data'][buffer_id] = loaded

        # NOTE: For eager mode, load micro batch is also eagerly executed if
        # next stage contains this instruction. So we also need to check
        # whether last stage is in r_stage.
        # FIXME(pengzhan): For lazy mode, load micro batch is not executed.
        # When failure happens, a new load micro batch instruction will be
        # inserted. But this instruction can only fetch the first batch. So
        # We need to offset the data loader.
        if self.num_stages - 1 in self.stage_ids or self.num_stages - 1 in self.r_stage_ids:
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
            elif isinstance(batch[1], tuple):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)

            self.pipe_buffers['label'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            send_bytes += _tensor_bytes(buffer)
        elif isinstance(buffer, list):
            assert (False)
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                send_bytes += _tensor_bytes(tensor)
        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for idx, tensor in enumerate(buffer):
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                # Useful for performance debugging.
                '''
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                '''
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            return self._allocate_buffer(recv_shape, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes = []
            for idx in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes.append(recv_shape.tolist())

            buffers = self._allocate_buffers(recv_shapes, num_buffers=1)[0]
            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id, stage_id):
        # Internal communication
        if self._inc(stage_id) in self.stage_ids:
            return

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        if buffer_id >= 0:
            outputs = self.pipe_buffers[f'output_{stage_id}'][buffer_id]
        else:
            outputs = self.ping_tensor

        if buffer_id >= 0:
            if self.first_output_send:
                self.first_output_send = False
                self._send_tensor_meta(outputs, self.next_stage)

        def send_handler(stage):
            if isinstance(outputs, torch.Tensor):
                p2p.send(outputs, stage)
            elif isinstance(outputs, tuple):
                for idx, buffer in enumerate(outputs):
                    p2p.send(buffer, stage)
            else:
                raise NotImplementedError('Could not send output of type '
                                          f'{type(outputs)}')

        try:
            send_handler(self.next_stage)
        except Exception as e:
            self.log(f"---- SEND ACTS FAILED!!!! RESORTING TO FALLBACK", color=True)
            failed_rank = self.grid._topo.get_rank(data=self.grid.get_data_parallel_id(), pipe=self.next_stage)
            self.global_store.set(str(failed_rank), '1')
            self.coordinates.append([self.grid.get_data_parallel_id(), self.next_stage])
            self.rdzv_handler.update_coordinates(self.global_rank, self.coordinates)
            raise NextStageException(e)

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()

    def _exec_send_grads(self, buffer_id, stage_id):
        # Internal communication
        if self._dec(stage_id) in self.stage_ids:
            return

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        use_grad_buffer = False
        input_key = f'input_grad_{stage_id}'
        if input_key in self.pipe_buffers and \
                self.pipe_buffers[input_key][buffer_id] is not None:
            inputs = self.pipe_buffers[input_key][buffer_id]\
                .to(self.device)
            use_grad_buffer = True
        else:
            inputs = self.pipe_buffers[f'input_{stage_id}'][buffer_id]
            assert inputs.grad is not None
            inputs = inputs.grad

        # XXX Terrible hack
        # Drop the attention mask from the input buffer here. It does not have
        # a grad that needs to be communicated. We free the buffer immediately
        # after, so no need to restore it. The receiver also has a hack that skips
        # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
        if self.module.__class__.__name__ == 'GPT2ModelPipe':
            inputs = list(inputs)
            inputs.pop()
            inputs = tuple(inputs)

        def send_handler(stage):
            if isinstance(inputs, torch.Tensor):
                p2p.send(inputs, stage)
            else:
                raise NotImplementedError()
                # NOTE: Temporarily disable for development
                # for idx, buffer in enumerate(inputs):
                #     # Skip tensors that will not produce a grad
                #     if not buffer.is_floating_point():
                #         assert buffer.grad is None
                #         continue
                #     assert buffer.grad is not None
                #     p2p.send(buffer.grad, stage)
        try:
            send_handler(self.prev_stage)
        except Exception as e:
            self.log(f"---- SEND GRADS FAILED!!!! RESORTING TO FALLBACK", color=True)
            failed_rank = self.grid._topo.get_rank(data=self.grid.get_data_parallel_id(), pipe=self.prev_stage)
            self.global_store.set(str(failed_rank), '1')
            raise PrevStageException(e)

        # Retire input tensor
        if self.redundancy_level > 0 and stage_id != 0:
            if use_grad_buffer:
                self.pipe_buffers[f'input_grad_{stage_id}'][buffer_id] = None
            else:
                self.pipe_buffers[f'input_grad_{stage_id}'][buffer_id] = \
                    inputs.clone().detach().to('cpu')
        else:
            self.pipe_buffers[f'input_{stage_id}'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()

    def _exec_recv_activations(self, buffer_id, stage_id):
        # Internal communication
        if  self._dec(stage_id) in self.stage_ids or (self.eager_recovery and stage_id in self.r_stage_ids):
            if self.pipe_buffers[f'input_{stage_id}'][buffer_id] is None:
                output = self.pipe_buffers[f'output_{self._dec(stage_id)}'][buffer_id].clone().detach().to(self.device)
                output.requires_grad = True
                self.pipe_buffers[f'input_{stage_id}'][buffer_id] = output
            return

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        # Allocate the buffer if necessary
        buffer = None
        if buffer_id >= 0:
            if self.pipe_recv_buf is None:
                self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)
            buffer = self.pipe_recv_buf
        else:
            buffer = self.ping_buffer

        def recv_handler(stage):
            if isinstance(buffer, torch.Tensor):
                p2p.recv(buffer, stage)
                recvd = buffer.clone().detach()
                recvd.requires_grad = recvd.is_floating_point()
            else:
                raise NotImplemented("Not support receiving tuple")

            return recvd
        try:
            recvd = recv_handler(self.prev_stage)
        except Exception as e:
            self.log(f"---- RECV ACTS FAILED!!!! RESORTING TO FALLBACK", color=True)
            failed_rank = self.grid._topo.get_rank(data=self.grid.get_data_parallel_id(), pipe=self.prev_stage)
            self.global_store.set(str(failed_rank), '1')
            raise PrevStageException(e)

        if buffer_id >= 0:
            self.pipe_buffers[f'input_{stage_id}'][buffer_id] = recvd

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()

    def _exec_recv_grads(self, buffer_id, stage_id):
        # Internal communicaiton
        if self._inc(stage_id) in self.stage_ids:
            self.grad_layer = \
                self.pipe_buffers[f'input_{self._inc(stage_id)}'][buffer_id].grad
            return

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers[f'output_{stage_id}'][buffer_id]

        # Allocate gradient if necessary
        if self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                self.grad_layer = self._allocate_buffer(s, num_buffers=1)[0]
            else:
                sizes = [list(t.size()) for t in outputs if t.is_floating_point()]
                self.grad_layer = self._allocate_buffers(sizes, num_buffers=1)[0]

        def recv_handler(stage):
            if isinstance(self.grad_layer, torch.Tensor):
                p2p.recv(self.grad_layer, stage)
            else:
                assert isinstance(outputs, tuple)
                for idx, buffer in enumerate(self.grad_layer):
                    p2p.recv(buffer, stage)
        try:
            recv_handler(self.next_stage)
        except Exception as e:
            self.log(f"---- RECV GRADS FAILED!!!! RESORTING TO FALLBACK", color=True)
            failed_rank = self.grid._topo.get_rank(data=self.grid.get_data_parallel_id(), pipe=self.next_stage)
            self.global_store.set(str(failed_rank), '1')
            self.coordinates.append([self.grid.get_data_parallel_id(), self.next_stage])
            self.rdzv_handler.update_coordinates(self.global_rank, self.coordinates)
            raise NextStageException(e)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()

    def _exec_send_weights(self, stage_id):
        # TODO(pengzhan): Implement send weights without p2p group

        src_rank = self.grid.stage_to_global(stage_id)
        for user_stage in self.r_user_stage_ids:
            dst_rank = self.grid.stage_to_global(user_stage)
            group = self.sync_group[src_rank][dst_rank]
            if group is None:
                raise Exception(
                    f'Group of {src_rank} to {dst_rank} does not exist')

            if not self.eager_recovery:
                for _, param in self.module.get_named_param(stage_id):
                    dist.send(param.detach().clone().to('cpu'),
                              dst_rank, group=group)
                for _, state in super().get_named_state(stage_id):
                    dist.send(state.detach().clone().to('cpu'),
                              dst_rank, group=group)
            else:
                for _, param in self.module.get_named_param(stage_id):
                    dist.broadcast(
                        param.detach(), self.global_rank, group=group)
                for _, state in super().get_named_state(stage_id):
                    dist.broadcast(
                        state.detach(), self.global_rank, group=group)

    def _exec_recv_weights(self, stage_id):
        # TODO(pengzhan): Implement recv weights without p2p group
        return
        dst_rank = self.grid.stage_to_global(self.stage_id)
        src_rank = self.grid.stage_to_global(stage_id)
        group = self.sync_group[src_rank][dst_rank]
        if group is None:
            raise Exception(
                f'Group of {src_rank} to {dst_rank} does not exist')

        self.recv_weights_work = []
        if not self.eager_recovery:
            for _, buffer in self.param_buffers[stage_id].items():
                work = dist.irecv(buffer, src_rank, group=group)
                self.recv_weights_work.append(work)
            for _, buffer in self.state_buffers[stage_id].items():
                work = dist.irecv(buffer, src_rank, group=group)
                self.recv_weights_work.append(work)
        else:
            for _, buffer in self.param_buffers[stage_id].items():
                work = dist.broadcast(
                    buffer, src_rank, group=group, async_op=True)
                self.recv_weights_work.append(work)
                # FIXME(pengzhan): Refresh param
            for _, buffer in self.state_buffers[stage_id].items():
                work = dist.broadcast(
                    buffer, src_rank, group=group, async_op=True)
                self.recv_weights_work.append(work)

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()
        self.mem_status('BEFORE STEP', reset_max=True)

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        # self.mem_status('AFTER STEP')

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/lr',
                                        self.get_lr()[0],
                                        self.global_samples)]
                if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                    self.summary_events.append((f'Train/Samples/loss_scale',
                                                self.optimizer.cur_scale,
                                                self.global_samples))
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input',
                    'forward_microstep',
                    'backward_microstep',
                    'backward_inner_microstep',
                    'backward_allreduce_microstep',
                    'backward_tied_allreduce_microstep',
                    'step_microstep'
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'step'
                ])

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, fp16=None, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            fp16 (bool): whether to use FP16. default: defer to self.fp16_enabled()
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """

        if fp16 is None:
            fp16 = self.fp16_enabled()

        if fp16:
            return torch.zeros(shape, dtype=torch.half, device=self.device, **kwargs)
        else:
            return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_buffers(self, shapes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for shape in shapes:
                buffer.append(self._allocate_zeros(shape, requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def mem_status(self, msg, reset_max=False):
        if not self.enable_mem_status:
            return

        global mem_alloced, mem_cached

        if self.global_steps != 1:
            return

        torch.cuda.synchronize()

        # NOTE: Temporarily disable for development
        # if reset_max:
        #     torch.cuda.reset_max_memory_cached()
        #     torch.cuda.reset_max_memory_allocated()

        new_alloced = torch.cuda.memory_allocated()
        new_cached = torch.cuda.memory_cached()

        delta_alloced = new_alloced - mem_alloced
        delta_cached = new_cached - mem_cached

        mem_cached = new_cached
        mem_alloced = new_alloced

        max_alloced = torch.cuda.max_memory_allocated()
        max_cached = torch.cuda.max_memory_cached()

        # convert to MB for printing
        new_alloced /= 1024**2
        new_cached /= 1024**2
        delta_alloced /= 1024**2
        delta_cached /= 1024**2
        max_alloced /= 1024**2
        max_cached /= 1024**2

        print(
            f'RANK={self.global_rank} STAGE={self.stage_id} STEP={self.global_steps} MEMSTATS',
            msg,
            f'current alloc={new_alloced:0.4f}MB (delta={delta_alloced:0.4f}MB max={max_alloced:0.4f}MB) '
            f'current cache={new_cached:0.4f}MB (delta={delta_cached:0.4f}MB max={max_cached:0.4f}MB)'
        )

    def module_state_dict(self):
        """Override hack to save a pipe model and return the directory path of the save.

        This method should only be called by DeepSpeed's ``save_checkpoint()``. The
        recommended way of saving a ``PipelineModule`` outside of ``save_checkpoint()``
        is ``save_state_dict()``.

        Returns:
            None
        """
        assert isinstance(self.module, PipelineModule)
        assert self._curr_ckpt_path is not None, \
            "PipelineEngine expects module_state_dict() to be called from save_checkpoint()"

        self.module.save_state_dict(self._curr_ckpt_path)
        return None

    def load_module_state_dict(self, state_dict, strict=True):
        """Override hack to instead use a directory path.

        This is important because pipeline models checkpoint by layer instead of rank.

        If ``state_dict`` is not ``None`` or a ``str``, we revert to ``super()`` expecting a ``dict``.

        Args:
            state_dict (str, None): unused
            strict (bool, optional): Strict state loading. Defaults to True.
        """
        if (state_dict is not None) and (not isinstance(state_dict, str)):
            super().load_module_state_dict(state_dict, strict)
            return

        self.module.load_state_dir(load_dir=self._curr_ckpt_path, strict=strict)

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
        schedule.SendWeights: _exec_send_weights,
        schedule.RecvWeights: _exec_recv_weights
    }

    def _exec_schedule(self, pipe_schedule, start_step=0, debug=False) -> Optional[Tuple[int, Exception]]:
        """ Execute schedule from `start_step`, and return failed step or None
        indicates success.
        """
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []

        # NOTE: Error handling mechanism
        # - Handle in step granularity, instead of instruction granularity.
        # - It is safe to re-execute instructions in current step if it is
        # a communication exception, because communication always happens
        # before computation.

        # For each step in the schedule
        exception_status = None
        for i, step_cmds in enumerate(pipe_schedule):
            if i < start_step:
                continue

            if debug:
                print(f'[DEBUG Pipeline] Instructions in step {i}:', end=' ')
                print(*step_cmds, sep=',')

            cmd_types = [type(cmd) for cmd in step_cmds]
            if schedule.ReduceGrads in cmd_types:
                for rank in self.grid.current_dp_group:
                    if int(self.global_store.get(str(rank))) == 1:
                        failed_data_parallel_id = self.grid._topo.get_coord(rank).data
                        self.log(f'RANK {rank} FROM PIPELINE {failed_data_parallel_id} IN ALL-REDUCE GROUP FAILED. USING FALLBACK', color=True)
                        e = AllReduceException(failed_data_parallel_id, f'RANK {rank} FROM PIPELINE {failed_data_parallel_id} IN ALL-REDUCE GROUP FAILED. USING FALLBACK')
                        msg = f'{type(cmd)}: {e}'
                        e = type(e)(e.src, msg)
                        exception_status = (i, e)
                        return exception_status

            # For each instruction in the step
            for cmd in step_cmds:
                try:
                    if type(cmd) not in self._INSTRUCTION_MAP:
                        raise RuntimeError(f'{self.__class__.__name__} does not understand instruction {repr(cmd)}')

                    self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    if debug: print(cmd)
                    self._exec_instr(**cmd.kwargs)
                except Exception as e:
                    msg = f'{type(cmd)}: {e}'
                    if hasattr(e, 'src'):
                        e = type(e)(e.src, msg)
                    else:
                        e = type(e)(msg)
                    exception_status = (i, e)
                    break

            if exception_status is not None:
                break
        return exception_status

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        self.batch_fn = fn
