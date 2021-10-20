from dataclasses import dataclass
# from typing import List, Tuple

from torch.distributed.elastic.agent.server import WorkerSpec

@dataclass
class ProjectPactumWorkerSpec(WorkerSpec):
    max_pipe_parallel_size: int = 1
