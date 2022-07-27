from typing import Callable, List, Tuple
from ..utils import call_to_str

from abc import ABC, abstractmethod

def _step_to_micro_batch(stages: int, stage_id: int, step_id: int) -> Tuple[int,bool]:
    def _even_step_forward_id(step_id):
        base = step_id // 2
        micro_batch_id = int(base - stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(step_id):
        base = step_id // 2
        micro_batch_id = int(base - stages + (stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(step_id):
        base = ((step_id - 1) // 2) - stages + 1
        micro_batch_id = int(base + stage_id // 2)
        return micro_batch_id

    if _is_even(step_id) and _is_even(stage_id):
        micro_batch_id = _even_step_forward_id(step_id)
        is_forward = True

    elif _is_odd(step_id) and _is_odd(stage_id):
        micro_batch_id = _odd_step_forward_id(step_id)
        is_forward = True

    elif _is_even(step_id) and _is_odd(stage_id):
        micro_batch_id = _even_step_backward_id(step_id)
        is_forward = False

    elif _is_odd(step_id) and _is_even(stage_id):
        micro_batch_id = _odd_step_backward_id(step_id)
        is_forward = False

    else:
        assert False

    return micro_batch_id, is_forward


class PipeSchedule(ABC):
    """Directs the execution of a pipeline engine by generating sequences of
    :class:`PipeInstruction`.

    Schedules are generators that yield sequences of
    :class:`PipeInstruction` to process the micro-batches in one batch.
    Each yielded step is atomic in the sense that a barrier
    synchronization can be placed between successive steps without
    deadlock.

    Below is an example schedule that implements data parallelism with gradient accumulation:

    .. code-block:: python

        class DataParallelSchedule(PipeSchedule):
            def steps(self):
                for step_id in range(self.micro_batches):
                    cmds = [
                        LoadMicroBatch(buffer_id=0),
                        ForwardPass(buffer_id=0),
                        BackwardPass(buffer_id=0),
                    ]
                    if step_id == self.micro_batches - 1:
                        cmds.extend([
                            ReduceGrads(),
                            OptimizerStep(),
                        ])
                    yield cmds

            def num_pipe_buffers(self):
                return 1

    Args:
        micro_batches (int): The number of micro-batches that comprise a batch.
        stages (int): The number of pipeline stages.
        stage_id (int): The pipe stage that will execute the generated schedule.
    """
    def __init__(self, micro_batches, stages, stage_id):
        super().__init__()
        self.micro_batches = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    @abstractmethod
    def steps(self):
        """Yield a list of :class:`PipeInstruction` for each step in the schedule.

        .. note::
            Schedules must implement ``steps()`` to define the schedule.

        Returns:
            Instructions to be executed as one step of the pipeline
        """
        pass

    def num_pipe_buffers(self):
        """The number of pipeline buffers that will be used by this stage.

        .. note::
            Schedules should specialize ``num_pipe_buffers()`` for memory savings at scale.

        Returns:
            The number of buffers for the engine to allocate.
        """
        return self.micro_batches

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.micro_batches

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages

    @property
    def stage(self):
        """Stage index used to configure this schedule."""
        return self.stage_id

    @property
    def num_stages(self):
        """The number of total pipeline stages used to configure this schedule."""
        return self.stages

    @property
    def num_micro_batches(self):
        """The number of total micro_batches used to configure this schedule."""
        return self.micro_batches

    @property
    def is_first_stage(self):
        """True if the configured ``stage_id`` is the first stage in the pipeline."""
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        """True if the configured ``stage_id`` is the last stage in the pipeline."""
        return self.stage_id == self.stages - 1

    def _buffer_idx(self, micro_batch_id):
        """Map a micro-batch index to a pipeline buffer index.

        This method uses a cyclic allocation strategy.

        Args:
            micro_batch_id (int): The micro-batch index relative to the beginning of the schedule.

        Returns:
            int: The index of the buffer that should store data.
        """
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % self.num_pipe_buffers()

    def __iter__(self):
        self.it = None
        return self

    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)


class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id

            # Alternate send/recv buffers
            if _is_even(self.stage_id):
                recv_buf = step_id % 2
                send_buf = (step_id + 1) % 2
            else:
                recv_buf = (step_id + 1) % 2
                send_buf = step_id % 2

            if self.is_first_stage or self.is_last_stage:
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(recv_buf))

            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(send_buf))
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))

                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(send_buf))

            if self._valid_micro_batch(micro_batch_id):
                cmds.append(ForwardPass(recv_buf))

            yield cmds

    def num_pipe_buffers(self):
        """Only two pipeline buffers are required for inferencing.

        Returns:
            ``2``
        """
        return 2


class TrainSchedule(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def __init__(self, micro_batches, stages, stage_id, **kwargs):
        super().__init__(micro_batches, stages, stage_id)

    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Load data
            if self.require_data_loader:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Exchange activations:
            # 1. recv act/grad for step computation
            # 2. send previous results ot neighboring nodes
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and \
                        self._valid_stage(self.prev_stage):
                    cmds.append(RecvActivation(curr_buffer, self.stage_id))
                if self._valid_micro_batch(prev_micro_batch_id) and \
                        self._valid_stage(self.prev_stage):
                    cmds.append(SendGrad(prev_buffer, self.stage_id))
            else:  # backward
                if self._valid_micro_batch(prev_micro_batch_id) and \
                        self._valid_stage(self.next_stage):
                    cmds.append(SendActivation(prev_buffer, self.stage_id))
                if self._valid_micro_batch(micro_batch_id) and \
                        self._valid_stage(self.next_stage):
                    cmds.append(RecvGrad(curr_buffer, self.stage_id))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer, self.stage_id))
                else:
                    cmds.append(BackwardPass(curr_buffer, self.stage_id))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                # NOTE: Temporarily disable for development
                # Enable it when running model with tied layers.
                # cmds.append(ReduceTiedGrads())

                # NOTE: Temporarily disable for development
                cmds.append(ReduceGrads(self.stage_id))
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    @property
    def require_data_loader(self):
        return self.is_first_stage or self.is_last_stage

    def num_pipe_buffers(self):
        # NOTE: To simplify failover, we make buffer id equal to micro batch id
        return super().num_pipe_buffers()

    def step_to_micro_batch(self, step_id:int) -> Tuple[int,bool]:
        return self._step_to_micro_batch(step_id)

    def valid_micro_batch(self, micro_batch_id:int) -> bool:
        return self._valid_micro_batch(micro_batch_id)

    def _step_to_micro_batch(self, step_id):
        return _step_to_micro_batch(self.stages, self.stage_id, step_id)


class ResilientPipeSchedule(ABC):
    def __init__(self, sched: TrainSchedule, next_sched: TrainSchedule, failed_step=0, curr_step=0):
        self.sched = sched
        self.next_sched = next_sched
        self.failed_step = failed_step
        self.curr_step = curr_step
        self.cmd_buffer = []

    def steps(self):
        for i, (step_cmds, next_step_cmds) in enumerate(zip(self.sched, self.next_sched)):
            yield self._merge_cmds(step_cmds, next_step_cmds, i)

    def num_pipe_buffers(self):
        # TODO(pengzhan): Calculate accurate buffer required
        return self.sched.num_pipe_buffers()

    @abstractmethod
    def _merge_cmds(self, cmds, next_cmds, i):
        pass

    def __iter__(self):
        self.it = None
        return self

    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)


class NextStageFailoverSchedule(ResilientPipeSchedule):

    def _merge_cmds(self, cmds, next_cmds, i):
        rets = []

        # Redo
        if i < self.curr_step:
            if i < self.failed_step:
                rets += [c for c in next_cmds if type(c) != SendActivation and type(c) != SendGrad]
            else:
                rets += next_cmds

        # Merge
        else:
            # TODO(pengzhan): Refactor: remove arguments for function
            def cmds_type(i):
                return type(cmds[i]) if i < len(cmds) else None

            def cmds_append(i):
                rets.append(cmds[i])
                return i+1

            def next_cmds_type(i):
                return type(next_cmds[i]) if i < len(next_cmds) else None

            def next_cmds_append(i, cmd=None):
                rets.append(next_cmds[i] if cmd is None else cmd)
                return i+1

            idx, next_idx = 0, 0
            while idx < len(cmds) or next_idx < len(next_cmds):
                if cmds_type(idx) == LoadMicroBatch:
                    idx = cmds_append(idx)

                elif cmds_type(idx) == RecvWeights:
                    idx += 1

                elif next_cmds_type(next_idx) == SendWeights:
                    next_idx += 1

                elif next_cmds_type(next_idx) == LoadMicroBatch:
                    next_idx = next_cmds_append(next_idx)

                elif next_cmds_type(next_idx) == SendActivation or next_cmds_type(next_idx) == RecvGrad:
                    next_idx = next_cmds_append(next_idx)

                elif next_cmds_type(next_idx) == SendGrad:
                    assert cmds_type(idx) == RecvGrad, \
                        f"Unable to merge cmds ({cmds}) and ({next_cmds})"
                    next_idx = next_idx + 1
                    idx = cmds_append(idx)

                elif next_cmds_type(next_idx) == RecvActivation:
                    assert cmds_type(idx) == SendActivation, \
                        f"Unable to merge cmds ({cmds}) and ({next_cmds})"
                    idx = idx + 1
                    next_idx = next_cmds_append(next_idx)

                elif cmds_type(idx) == BackwardPass:
                    idx = cmds_append(idx)

                elif next_cmds_type(next_idx) == BackwardPass:
                    next_idx = next_cmds_append(next_idx)

                elif cmds_type(idx) is not None:
                    idx = cmds_append(idx)

                elif next_cmds_type(next_idx) is not None:
                    if next_cmds_type(next_idx) == ForwardPass or \
                            next_cmds_type(next_idx) == BackwardPass or \
                            next_cmds_type(next_idx) == RecvWeights:
                        next_idx = next_cmds_append(next_idx)
                    # Skip optimizer step as a single optimizer step will
                    # handle two stages.
                    elif next_cmds_type(next_idx) == OptimizerStep:
                        next_idx += 1
                    # Still need to do reduce grad for two stages.
                    elif next_cmds_type(next_idx) == ReduceGrads:
                        next_idx = next_cmds_append(next_idx)
                    else:
                        msg = f"Unsupported cmd found in cmds ({next_cmds})"
                        raise Exception(msg)

                else:
                    msg = f"Unable to merge cmds ({cmds}) and ({next_cmds})"
                    raise Exception(msg)

        return rets


class PrevStageFailoverSchedule(ResilientPipeSchedule):

    def _merge_cmds(self, cmds, next_cmds, i):
        rets = []

        # Redo
        if i < self.curr_step:
            if i < self.failed_step:
                rets += [c for c in next_cmds if type(c) == SendGrad]

        # Merge
        else:
            for next_cmd in next_cmds:
                # Disable recursive recovery. See FIXME in
                # PipelineEngine.train_batch
                if type(next_cmd) == SendWeights:
                    continue
                if type(next_cmd) == RecvActivation and next_cmd.buffer_id == -1:
                    continue
                rets.append(next_cmd)

        return rets


class AllReduceFailoverSchedule(ResilientPipeSchedule):
    def _merge_cmds(self, cmds, next_cmds, i):
        rets = []

        if i < self.failed_step:
            pass
        else:
            rets = cmds

        return rets


class LazyRecoverySchedule(ResilientPipeSchedule):

    def __init__(self, sched, next_sched, failed_step=0, curr_step=0):
        super().__init__(sched, next_sched, failed_step, curr_step)
        self.stage_id: int = self.sched.stage_id

        # Wrapper for TrainSchedule
        self.stage_id: int = self.sched.stage_id
        self.is_last_stage: bool = self.sched.is_last_stage
        self.is_first_stage: bool = self.sched.is_first_stage
        self.is_forward: Callable[[int,int],bool] = \
            lambda stage_id, step_id: _step_to_micro_batch(self.sched.stages, stage_id, step_id)[1]
        self.contain_communication: Callable[[List[PipeInstruction]],bool] = \
            lambda cmds: any([isinstance(c, CommunicationInstruction) for c in cmds])
        self.is_first_half: Callable[[int],bool] = \
            lambda i: i < (self.sched.micro_batches + self.sched.stages - 1)

    def _merge_cmds(self, cmds, next_cmds, i):
        if self.is_first_stage:
            if self.is_forward(self.stage_id, i):
                cmds = [RecvActivation(-1, self.stage_id)] + cmds
        elif self.is_last_stage:
            if not self.contain_communication(cmds) and self.is_forward(self.stage_id, i) and self.is_first_half(i):
                cmds = [RecvActivation(-1, self.stage_id)] + cmds
            if self.is_forward(0, i):
                cmds = [SendActivation(-1, self.stage_id)] + cmds
        else:
            if not self.contain_communication(cmds) and self.is_first_half(i):
                if not self.is_forward(self.stage_id, i):
                    cmds = [SendActivation(-1, self.stage_id)] + cmds
                else:
                    cmds = [RecvActivation(-1, self.stage_id)] + cmds
        return cmds


class EagerRecoverySchedule(ResilientPipeSchedule):
    def _merge_cmds(self, cmds, next_cmds, i):
        rets = []

        redundant_instructions = [RecvActivation, LoadMicroBatch, ForwardPass]

        def append_next_cmds(next_cmds):
            for cmd in next_cmds:
                if type(cmd) in redundant_instructions:
                    rets.append(cmd)

        # Always insert redundant computations before RecvGrad
        if RecvGrad in [type(c) for c in cmds] and len(self.cmd_buffer) > 0:
            append_next_cmds(self.cmd_buffer.pop(0))

        if any([type(i) in redundant_instructions for i in next_cmds]):
            self.cmd_buffer.append(next_cmds)

        for cmd in cmds:
            rets.append(cmd)
        return rets


class DataParallelSchedule(PipeSchedule):
    """An example schedule that trains using traditional data parallelism with gradient
    accumulation.
    """
    def steps(self):
        """"""
        for step_id in range(self.micro_batches):
            cmds = [
                LoadMicroBatch(buffer_id=0),
                ForwardPass(buffer_id=0),
                BackwardPass(buffer_id=0),
            ]
            if step_id == self.micro_batches - 1:
                cmds.extend([
                    ReduceGrads(),
                    OptimizerStep(),
                ])
            yield cmds

    def num_pipe_buffers(self):
        """Only one pipeline buffer needed.
        """
        return 1


class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.

    All keyword arguments are stored as members similar to a ``namedtuple``. These are
    then accessible to the :class:`PipeEngine` during execution.

    Args:
        kwargs (optional): keyword arguments to store as members
    """
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return call_to_str(self.name, **self.kwargs)


class BufferOpInstruction(PipeInstruction):
    """A pipeline instruction that operates on pipeline buffer(s).

    Args:
        buffer_id (int): the index of the pipeline buffer() to modify.
    """
    def __init__(self, buffer_id, **kwargs):
        super().__init__(buffer_id=buffer_id, **kwargs)


class MultiStageOpInstruction(PipeInstruction):
    def __init__(self, stage_id, **kwargs):
        super().__init__(stage_id=stage_id, **kwargs)


class MultiStageBufferOpInstruction(BufferOpInstruction):
    def __init__(self, buffer_id, stage_id, **kwargs):
        super().__init__(buffer_id=buffer_id, stage_id=stage_id, **kwargs)


# IO
class LoadMicroBatch(BufferOpInstruction):
    """Load a micro-batch into a buffer.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = next(data_iter)
    """
    pass


# Compute
class ForwardPass(MultiStageBufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['ouputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass


class BackwardPass(MultiStageBufferOpInstruction):
    """Compute a backward pass and accumulate gradients.

    Roughly:

    .. code-block:: python

        outputs = buffers['ouputs'][buffer_id]
        gradients = buffers['gradients'][buffer_id]
        torch.autograd.backward(tensors=outputs,
                                grad_tensors=gradients)
    """
    pass


# Communication
class CommunicationInstruction:
    ...


class SendActivation(MultiStageBufferOpInstruction, CommunicationInstruction):
    """Send activations to the next stage in the pipeline.

    Roughly:

    .. code-block:: python

        send(buffers['outputs'][buffer_id])

    .. note::
        The communication is blocking and must be paired with a :class:`RecvActivation`
        on the next pipeline stage to avoid deadlock.
    """
    pass


class RecvActivation(MultiStageBufferOpInstruction, CommunicationInstruction):
    """Receive activations from the previous stage in the pipeline.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = recv()

    .. note::
        The communication is blocking and must be paired with a :class:`SendActivation`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class SendGrad(MultiStageBufferOpInstruction, CommunicationInstruction):
    """Send computed gradients to the previous pipeline stage.
    with respect to the received activations

    .. note::
        Only received tensors with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None`` on the receiving stage.

    .. note::
        The communication is blocking and must be paired with a :class:`RecvGrad`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class RecvGrad(MultiStageBufferOpInstruction, CommunicationInstruction):
    """Receive computed gradients the next pipeline stage.

    .. note::
        Only activations with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None``.

    .. note::
        The communication is blocking and must be paired with a :class:`SendGrad`
        on the next pipeline stage to avoid deadlock.
    """
    pass


# Other
class SendWeights(MultiStageOpInstruction):
    pass


class RecvWeights(MultiStageOpInstruction):
    pass


class OptimizerStep(PipeInstruction):
    """Performs one step with the optimizer and zeros gradients.

    .. note:: Should be issued after :class:`ReduceGrads` and :class:`ReduceTiedGrads`.

    .. note:: Can be a synchronization point among data-parallel ranks.
    """
    pass


class ReduceGrads(MultiStageOpInstruction):
    """Reduce the computed gradients among data-parallel processes within the stage.
    """
    pass


class ReduceTiedGrads(PipeInstruction):
    """Reduce the computed gradients of tied modules within a pipeline-parallel group.

    .. warning::
        The stages included in this synchronization point are not known until
        the model is partitioned among pipeline stages. In the worst case, it
        includes all pipeline stages. This instruction should be scheduled
        carefully to avoid deadlocks.
    """
    pass


def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0
