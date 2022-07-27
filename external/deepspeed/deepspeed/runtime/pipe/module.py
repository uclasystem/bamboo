import collections
import os
import glob
import enum
import time
from pathlib import Path

import re as regex

from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parameter import Parameter

from deepspeed.utils import logger
from .. import utils as ds_utils
from ..activation_checkpointing import checkpointing
from .topology import PipeDataParallelTopology, PipelineParallelGrid
from deepspeed.runtime.state_dict_factory import SDLoaderFactory

from typing import Dict, List, Callable


class PipelineError(Exception):
    """Errors related to the use of deepspeed.PipelineModule """


class LayerSpec:
    """Building block for specifying pipeline-parallel modules.

    LayerSpec stores the type information and parameters for each stage in a
    PipelineModule. For example:

    .. code-block:: python

        nn.Sequence(
            torch.nn.Linear(self.in_dim, self.hidden_dim, bias=False),
            torch.nn.Linear(self.hidden_hidden, self.out_dim)
        )

    becomes

    .. code-block:: python

        layer_specs = [
            LayerSpec(torch.nn.Linear, self.in_dim, self.hidden_dim, bias=False),
            LayerSpec(torch.nn.Linear, self.hidden_hidden, self.out_dim)]
        ]
    """
    def __init__(self, typename, *module_args, **module_kwargs):
        self.typename = typename
        self.module_args = module_args
        self.module_kwargs = module_kwargs

        if not issubclass(typename, nn.Module):
            raise RuntimeError('LayerSpec only supports torch.nn.Module types.')

        if dist.is_initialized():
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = -1

    def __repr__(self):
        return ds_utils.call_to_str(self.typename.__name__,
                                    self.module_args,
                                    self.module_kwargs)

    def build(self, log=False):
        """Build the stored specification."""
        if log:
            logger.info(f'RANK={self.global_rank} building {repr(self)}')

        return self.typename(*self.module_args, **self.module_kwargs)


class TiedLayerSpec(LayerSpec):
    def __init__(self,
                 key,
                 typename,
                 *module_args,
                 forward_fn=None,
                 tied_weight_attr='weight',
                 **module_kwargs):
        super().__init__(typename, *module_args, **module_kwargs)
        self.key = key
        self.forward_fn = forward_fn
        self.tied_weight_attr = tied_weight_attr



class PipelineModule(nn.Module):
    def __init__(self,
                 layers,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seed_layers=False,
                 seed_fn=None,
                 base_seed=1234,
                 partition_method='parameters',
                 custom_partitions=[],
                 activation_checkpoint_interval=0,
                 activation_checkpoint_func=checkpointing.checkpoint):
        """Modules to be parallelized with pipeline parallelism.

        The key constraint that enables pipeline parallelism is the
        representation of the forward pass as a sequence of layers
        and the enforcement of a simple interface between them. The
        forward pass is implicitly defined by the module ``layers``. The key
        assumption is that the output of each layer can be directly fed as
        input to the next, like a ``torch.nn.Sequence``. The forward pass is
        implicitly:

        .. code-block:: python

            def forward(self, inputs):
                x = inputs
                for layer in self.layers:
                    x = layer(x)
                return x

        .. note::
            Pipeline parallelism is not compatible with ZeRO-2 and ZeRO-3.

        Args:
            layers (Iterable): A sequence of layers defining pipeline structure. Can be a ``torch.nn.Sequential`` module.
            num_stages (int, optional): The degree of pipeline parallelism. If not specified, ``topology`` must be provided.
            topology (``deepseed.pipe.ProcessTopology``, optional): Defines the axes of parallelism axes for training. Must be provided if ``num_stages`` is ``None``.
            loss_fn (callable, optional): Loss is computed ``loss = loss_fn(outputs, label)``
            base_seed (int, optional): [description]. Defaults to 1234.
            partition_method (str, optional): [description]. Defaults to 'parameters'.
            custom_partitions (Iterable, optional):  Manually specify partition boundaries.
            activation_checkpoint_interval (int, optional): The granularity activation checkpointing in terms of number of layers. 0 disables activation checkpointing.
            activation_checkpoint_func (callable, optional): The function to use for activation checkpointing. Defaults to ``deepspeed.checkpointing.checkpoint``.
        """

        super().__init__()

        if num_stages is None and topology is None:
            raise RuntimeError('must provide num_stages or topology')

        self.micro_offset = 0

        self.loss_fn = loss_fn

        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(
                f'SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}'
            )

        # Setup world info
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        assert self.local_rank is not None

        if topology:
            self._topo = topology
            self.num_stages = self._topo.get_dim('pipe')
        else:
            self.num_stages = num_stages
            if topology is None:
                if self.world_size % self.num_stages != 0:
                    raise RuntimeError(
                        f'num_stages ({self.num_stages}) must divide distributed world size ({self.world_size})'
                    )
                dp = self.world_size // num_stages
                topology = PipeDataParallelTopology(num_pp=num_stages, num_dp=dp)
                self._topo = topology

        # Contruct communicators for pipeline topology
        self._grid = PipelineParallelGrid(process_group=self.world_group,
                                          topology=self._topo)

        self.stage_id = self._topo.get_coord(self.global_rank).pipe

        # Initialize partition information
        self.partition_method = partition_method
        self._layer_specs = list(layers)
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=partition_method, partitions=custom_partitions)

        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}

        # Offset the random seed by the stage ID.
        #newseed = torch.cuda.initial_seed() + self._grid.get_stage_id()
        #ds_utils.set_random_seed(newseed)

        #with torch.random.fork_rng(devices=[torch.cuda.current_device()]):

        # Use different buckets to record parameter names
        self.param_name_buckets: Dict[int, List[str]] = \
            collections.defaultdict(list)
        self.func_buckets: Dict[int, List[Callable]] = \
            collections.defaultdict(list)

        self.build_layers(self.stage_id)

        self.tied_comms = self._index_tied_modules()
        self._synchronize_tied_weights()

        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.activation_checkpoint_func = activation_checkpoint_func

    def get_named_param(self, stage_id):
        for name, param in self.named_parameters():
            if name not in self.param_name_buckets[stage_id]:
                continue
            yield name, param

    def allocate_param(self, stage_id, param_buffer, device='cpu'):
        specs = self._layer_specs
        local_start, local_stop = self.parts[stage_id], self.parts[stage_id+1]

        def add_to_buffer(name, layer):
            for n, p in layer.named_parameters():
                if not p.requires_grad:
                    continue
                param_buffer[f'{name}.{n}'] = torch.zeros_like(p).to(device)

        param_names = []
        for local_idx, layer in enumerate(specs[local_start:local_stop]):
            layer_idx = local_idx + local_start
            if isinstance(layer, PipelineModule):
                raise NotImplementedError('RECURSIVE BUILD NOT YET IMPLEMENTED')

            elif isinstance(layer, nn.Module):
                name = str(layer_idx)
                param_names += [f'{name}.{n}'
                                for n, p in layer.named_parameters()
                                if p.requires_grad]
                add_to_buffer(name, layer)

            elif isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)
                param_names += [f'{name}.{n}'
                                for n, p in layer.named_parameters()
                                if p.requires_grad]
                add_to_buffer(name, module)

            else:
                continue

        self.param_name_buckets[stage_id] = param_names
        return param_names

    def build_layers(self, stage_id, buffer=None):
        specs = self._layer_specs
        local_start, local_stop = self.parts[stage_id], self.parts[stage_id+1]
        funcs = self.func_buckets[stage_id]

        param_names = []
        for local_idx, layer in enumerate(specs[local_start:local_stop]):
            layer_idx = local_idx + local_start
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    ds_utils.set_random_seed(self.base_seed + layer_idx)

            # Recursively build PipelineModule objects
            if isinstance(layer, PipelineModule):
                raise NotImplementedError('RECURSIVE BUILD NOT YET IMPLEMENTED')

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, nn.Module):
                name = str(layer_idx)
                funcs.append(layer)
                param_names += [f'{name}.{n}'
                                for n, p in layer.named_parameters()
                                if p.requires_grad]
                self.add_module(name, layer)

            # TiedLayerSpec objects contain an nn.Module that should be allocated now.
            # TODO(pengzhan): Get parameter of new added tied layer spec
            elif isinstance(layer, TiedLayerSpec):
                # Build and register the module if we haven't seen it before.
                if layer.key not in self.tied_modules:
                    self.tied_modules[layer.key] = layer.build()
                    self.tied_weight_attrs[layer.key] = layer.tied_weight_attr

                if layer.forward_fn is None:
                    # Just use forward()
                    funcs.append(self.tied_modules[layer.key])
                else:
                    # User specified fn with args (module, input)
                    funcs.append(partial(layer.forward_fn, self.tied_modules[layer.key]))

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)
                funcs.append(module)
                if hasattr(layer, 'parameters'):
                    param_names += [f'{name}.{i}'
                                    for i, p in enumerate(module.parameters())
                                    if p.requires_grad]
                self.add_module(name, module)

            # Last option: layer may be a functional (e.g., lambda). We do nothing in
            # that case and just use it in forward()
            else:
                funcs.append(layer)

        self.param_name_buckets[stage_id] = param_names

        if buffer is not None:
            # NOTE: reload state dict does not work here. For an unknown result
            # reloading state dict changes the state of existing tensor.
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in buffer:
                        param.copy_(buffer[name])
            self.to(f'cuda:{self.local_rank}', non_blocking=True)
        else:
            self.to(f'cuda:{self.local_rank}')
        return param_names

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self._layer_specs)
        for idx, layer in enumerate(self._layer_specs):
            if isinstance(layer, LayerSpec):
                l = layer.build()
                params = filter(lambda p: p.requires_grad, l.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
            elif isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def _find_layer_type(self, layername):
        idxs = []
        typeregex = regex.compile(layername, regex.IGNORECASE)
        for idx, layer in enumerate(self._layer_specs):
            name = None
            if isinstance(layer, LayerSpec):
                name = layer.typename.__name__
            elif isinstance(layer, nn.Module):
                name = layer.__class__.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    continue
            if typeregex.search(name):
                idxs.append(idx)

        if len(idxs) == 0:
            raise RuntimeError(
                f"Partitioning '{layername}' found no valid layers to partition.")
        return idxs

    def print_layer_signatures(self):
        for stage, bucket in self.func_buckets.items():
            start = self.parts[stage]
            for i, layer in enumerate(bucket):
                if hasattr(layer, 'sum_params'):
                    print(f'LAYER {start + i}: {layer.sum_params()}')

    def load_layers(self, received_state, prev_model_state={}):
        bucket = self.func_buckets[self.stage_id]
        for i, l_id in enumerate(range(self.parts[self.stage_id], self.parts[self.stage_id + 1])):
            if hasattr(bucket[i], 'load_state_dict'):
                if l_id in prev_model_state:
                    lyr_state_dict = prev_model_state[l_id][0]
                    bucket[i].load_state_dict(lyr_state_dict)
                    continue

                if l_id in received_state:
                    lyr_state_dict = received_state[l_id][0]
                    bucket[i].load_state_dict(lyr_state_dict)
                    continue
            else:
                continue

    def get_layers(self, src_stage, layer_idxs):
        """ Return a list of all the layers specified in layer_idxs

        Args:
            layer_idxs (list): Global layer ids of layers to move

        Returns:
            list: layers specified in layer_idxs
        """
        layers = []
        part_start = self.parts[src_stage]
        funcs = self.func_buckets.get(src_stage, None)
        for idx in layer_idxs:
            layers.append(funcs[idx - part_start])

        return layers

    def move_between_stages(self, stage, layer_idxs):
        part_start = self.parts[stage]
        stage_funcs = self.func_buckets.get(stage, None)
        my_funcs = self.func_buckets.get(self.stage_id, None)
        for idx in layer_idxs:
            my_funcs.append(stage_funcs[idx - part_start])

    def remove_layers(self, stage, layer_idxs):
        """ Remove the layers that are specified by layer_idxs

        Args:
            layer_idxs (list): Global layer ids of layers to move
        """
        start = self.parts[stage]
        funcs = self.func_buckets.get(stage, None)
        for i in range(len(layer_idxs)-1, -1, -1):
            idx = layer_idxs[i]
            del funcs[idx - start]
            del self._modules[f'{idx}']

    def add_layers(self, layer_idxs, layer_state_dicts):
        """ Given a set of layer ids load them from memory (self._layer_specs)
            and then load their state dict

        Args:
            layer_idxs (list): Global layer ids of layers to move
            layer_state_dicts (list): State dicts for the layers containing updated
                parameters

        Returns:
            list: References to the udpated layers
        """
        ## Filter layers that do not have a state dict and check to make sure that
        ## it is equal
        layer_idxs = [idx for idx in layer_idxs if hasattr(self._layer_specs[idx], 'state_dict')]
        assert len(layer_idxs) == len(layer_state_dicts)

        layers = []
        insert_index = 0
        append_len = 0
        funcs = self.func_buckets.get(self.stage_id, None)
        for i, idx in enumerate(layer_idxs):
            if idx < self._local_start:
                lyr = self._layer_specs[idx]
                lyr = lyr.cuda()
                if hasattr(lyr, 'sum'):
                    print(f'LAYER {idx} BEFORE LOAD {lyr.sum()}')
                lyr.load_state_dict(layer_state_dicts[i])
                if hasattr(lyr, 'sum'):
                    print(f'LAYER {idx} AFTER LOAD {lyr.sum()}')
                    print()


                funcs.insert(insert_index, lyr)
                insert_index += 1

                layers.append(lyr)
                self.add_module(f'{idx}', lyr)
            elif idx >= self._local_stop:
                lyr = self._layer_specs[idx]
                lyr = lyr.cuda()
                if hasattr(lyr, 'sum'):
                    print(f'LAYER {idx} BEFORE LOAD {lyr.sum()}')
                lyr.load_state_dict(layer_state_dicts[i])
                if hasattr(lyr, 'sum'):
                    print(f'LAYER {idx} AFTER LOAD {lyr.sum()}')
                    print()

                funcs.append(lyr)
                append_len += 1

                layers.append(lyr)
                self.add_module(f'{idx}', lyr)
            else:
                lyr = funcs[idx - self._local_start + insert_index]
                if hasattr(lyr, 'sum'):
                    print(f'LAYER {idx} BEFORE LOAD {lyr.sum()}')
                lyr.load_state_dict(layer_state_dicts[i])
                if hasattr(lyr, 'sum'):
                    print(f'LAYER {idx} AFTER LOAD {lyr.sum()}')
                    print()
                layers.append(lyr)

        self._local_start -= insert_index
        self._local_stop += append_len

        return layers

    def reset_func_buckets(self, old_stage_id, new_parts):
        funcs = self.func_buckets.get(old_stage_id, None)
        self.func_buckets.clear()

        print(f'LEN FUNC BUCKETS {len(funcs)}')

        self.stage_id = self._topo.get_coord(self.global_rank).pipe
        self.func_buckets[self.stage_id] = funcs

        self.parts = new_parts
        self._local_start = new_parts[self.stage_id]
        self._local_stop = new_parts[self.stage_id + 1]

    def forward(self, forward_input, stage_id):
        # We need to offset the seed by the microbatch ID. Save it in a local var to
        # ensure it is preserved in the closure. Otherwise checkpointed forward funcs
        # will see a different offset.
        self.micro_offset += 1

        forward_funcs = self.func_buckets.get(stage_id, None)
        if forward_funcs is None:
            raise Exception(f'Unregistered stage id {stage_id}')

        def exec_range_func(start, end):
            ''' Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            '''
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = (self.base_seed *
                                    local_micro_offset) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            ds_utils.set_random_seed(new_seed)

                    inputs = layer(inputs)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(forward_funcs))
            x = func(forward_input)
        else:
            num_layers = len(forward_funcs)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(start_idx + self.activation_checkpoint_interval,
                              num_layers)

                funcs = forward_funcs[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x, )

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(
                        exec_range_func(start_idx,
                                        end_idx),
                        *x)
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x

    def _partition_layers(self, method='uniform', partitions=[]):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages with method {method}')

        method = method.lower()

        if method == 'custom':
            assert partitions, "Required customized partitions"
            self.parts = partitions

        elif method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(num_items=num_layers,
                                                    num_parts=num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(weights=param_counts,
                                                     num_parts=num_stages)
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            else:
                self.parts = ds_utils.partition_balanced(weights=binary_weights,
                                                         num_parts=num_stages)
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            print(f'parts={self.parts}')
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    print(f'    {idx+start:2d}: {name}')
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

    def get_new_partition(self, new_num_stages):
        method = self.partition_method.lower()

        # Each stage gets a simple uniform number of layers.
        parts = None
        if method == 'uniform':
            num_layers = len(self._layer_specs)
            parts = ds_utils.partition_uniform(num_items=num_layers,
                                                    num_parts=new_num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            parts = ds_utils.partition_balanced(weights=param_counts,
                                                     num_parts=new_num_stages)
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            else:
                parts = ds_utils.partition_balanced(weights=binary_weights,
                                                         num_parts=new_num_stages)
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        return parts

    def allreduce_tied_weight_gradients(self):
        '''All reduce the gradients of the tied weights between tied stages'''
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            dist.all_reduce(weight.grad, group=comm['group'])

    def _synchronize_tied_weights(self):
        for key, comm in self.tied_comms.items():
            dist.broadcast(
                getattr(comm['module'],
                        comm['weight_attr']),
                src=min(comm['ranks']),
                group=comm['group'],
            )

    def _index_tied_modules(self):
        ''' Build communication structures for tied modules. '''
        tied_comms = {}
        if self._topo.get_dim('pipe') == 1:
            return tied_comms

        specs = self._layer_specs
        tie_keys = set(s.key for s in specs if isinstance(s, TiedLayerSpec))
        for key in tie_keys:
            # Find the layers that the tied module appears in
            tied_layers = []
            for idx, layer in enumerate(specs):
                if isinstance(layer, TiedLayerSpec) and layer.key == key:
                    tied_layers.append(idx)
            # Find all stages with this tied module
            # TODO: Would be nice to remove the nested data/model parallelism loops and
            # TODO: instead generalize in some way, since we really just care about the
            # TODO: stage that owns the tied layer. Then loop over each (dp, mp, ...)
            # TODO: fiber to generate process groups.
            tied_stages = set(self.stage_owner(idx) for idx in tied_layers)
            for dp in range(self._grid.data_parallel_size):
                for mp in range(self._grid.get_slice_parallel_world_size()):
                    tied_ranks = []
                    for s in sorted(tied_stages):
                        if self._grid.get_slice_parallel_world_size() > 1:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp,
                                                           model=mp))
                        else:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp))
                    group = dist.new_group(ranks=tied_ranks)

                    # Record this tied module if we own a local copy of it.
                    if self.global_rank in tied_ranks:
                        assert key in self.tied_modules
                        if key in self.tied_modules:
                            tied_comms[key] = {
                                'ranks': tied_ranks,
                                'group': group,
                                'weight_attr': self.tied_weight_attrs[key],
                                'module': self.tied_modules[key],
                            }
                            # Only count the tied module once in the eyes of the FP16 optimizer
                            if self.global_rank != tied_ranks[0]:
                                for p in self.tied_modules[key].parameters():
                                    p.model_parallel = False
        '''
        if len(tied_comms) > 0:
            print(f'RANK={self.global_rank} tied_comms={tied_comms}')
        '''

        return tied_comms

    def partitions(self):
        return self.parts

    def stage_owner(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers
        for stage in range(self._topo.get_dim('pipe')):
            if self.parts[stage] <= layer_idx < self.parts[stage + 1]:
                return stage
        raise RuntimeError(f'Layer {layer_idx} not owned? parts={self.parts}')

    def _set_bounds(self, start=None, stop=None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive and stop is
        exclusive. The default of None for both results in all layers being built
        locally.
        """
        self._local_start = start
        self._local_stop = stop

    def set_checkpoint_interval(self, interval):
        assert interval >= 0
        self.checkpoint_interval = interval

    def topology(self):
        """ ProcessTopology object to query process mappings. """
        return self._topo

    def mpu(self):
        return self._grid

    def num_pipeline_stages(self):
        return self._topo.get_dim('pipe')

    def ckpt_prefix(self, checkpoints_path, tag):
        """Build a prefix for all checkpoint files written by this module. """
        # All checkpoint files start with this
        rank_name = 'module'

        # Data parallelism is omitted from the naming convention because we are agnostic
        # to this in the checkpoint.
        omit_dims = frozenset(['data'])
        axes = [a for a in self._grid._topo.get_axis_names() if a not in omit_dims]
        for dim in axes:
            rank = getattr(self._grid._topo.get_coord(rank=self.global_rank), dim)
            rank_name += f'-{dim}_{rank:02d}'

        ckpt_name = os.path.join(checkpoints_path, str(tag), rank_name)
        return ckpt_name

    def ckpt_layer_path(self, ckpt_dir, local_layer_idx):
        """Customize a prefix for a specific pipeline module layer. """
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f'layer_{idx:02d}')
        rank_repr = self._grid._topo.get_rank_repr(rank=self.global_rank)
        if rank_repr != '':
            layer_ckpt_path += f'-{rank_repr}'
        layer_ckpt_path += '-model_states.pt'
        return layer_ckpt_path

    def ckpt_layer_path_list(self, ckpt_dir, local_layer_idx):
        """Get all ckpt file list for a specific pipeline module layer. """
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f'layer_{idx:02d}')
        layer_ckpt_path += "*model_states.pt"
        ckpt_files = glob.glob(layer_ckpt_path)
        ckpt_files.sort()
        return ckpt_files

    def save_state_dict(self, save_dir):
        if self._grid.data_parallel_id != 0:
            return

        os.makedirs(save_dir, exist_ok=True)
        layer_offset = self._local_start
        for idx, layer in enumerate(self.func_buckets[self.stage_id]):
            model_ckpt_path = self.ckpt_layer_path(save_dir, idx)
            if not hasattr(layer, 'state_dict'):
                continue
            # We pass cloned tensors to torch.save() to avoid checkpoint bloat which occurs because torch.save()
            # saves the underlying storage rather than the slice of the storage corresponding to individual tensors.
            # This is a problem in DeepSpeed because we often allocate tensors using slices of large flattened buffers.
            # Tensor cloning helps to avoid this problem because the storage of cloned tensors are closer to the true size.
            # It is expected that the garbage collector will reclaim the cloned tensor storage to avoid memory bloat.
            # See https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
            orig_state_dict = layer.state_dict()
            final_state_dict = type(orig_state_dict)(
                {k: v.clone()
                 for k,
                 v in orig_state_dict.items()})
            torch.save(final_state_dict, model_ckpt_path)

    def load_state_dir(self, load_dir, strict=True):
        for idx, layer in enumerate(self.func_buckets[self.stage_id]):
            # Functions, etc. will not have state_dicts
            if not hasattr(layer, 'load_state_dict'):
                continue

            # get all checkpoint files for the layer.
            model_ckpt_list = self.ckpt_layer_path_list(load_dir, idx)
            mp_rank = self._grid.get_slice_parallel_rank()
            mp_world_size = self._grid.get_slice_parallel_world_size()

            sd_loader = SDLoaderFactory.get_sd_loader(model_ckpt_list, version=2.0)
            load_path, checkpoint, _ = sd_loader.load(mp_world_size, mp_rank, module_key=None, is_pipe_parallel=True)

            layer.load_state_dict(checkpoint)

            if self._grid.data_parallel_id == 0:
                logger.info(
                    f'RANK={self.global_rank} Loaded layer={idx+self._local_start} file={load_path}'
                )

        self._synchronize_tied_weights()

    def _is_checkpointable(self, funcs):
        if self.__class__.__name__ == 'GPT2ModelPipe':
            return all('ParallelTransformerLayerPipe' in f.__class__.__name__
                       for f in funcs)

        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)
