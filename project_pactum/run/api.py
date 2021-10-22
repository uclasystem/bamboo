import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, cast, Tuple

import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic import events, metrics
from torch.distributed.elastic.agent.server.api import WorkerSpec, WorkerState  # type: ignore[import]
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, record
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint
from torch.distributed.elastic.utils.logging import get_logger

logger = get_logger()

@dataclass
class ProjectPactumLaunchConfig:
    """
    min_nodes: Minimum amount of nodes that the user function will
                     be launched on. Elastic agent ensures that the user
                     function start only when the min_nodes amount enters
                     the rendezvous.
    max_nodes: Maximum amount of nodes that the user function
                     will be launched on.
    nproc_per_node: On each node the elastic agent will launch
                          this amount of workers that will execute user
                          defined function.
    rdzv_backend: rdzv_backend to use in the rendezvous (zeus-adapter, etcd).
    rdzv_endpoint: The endpoint of the rdzv sync. storage.
    rdzv_id: The unique run id of the job (if not passed a unique one will be
             deduced from run environment - flow workflow id in flow - or auto generated).
    role: User defined role of the worker (defaults to "trainer").
    max_restarts: The maximum amount of restarts that elastic agent will conduct
                  on workers before failure.
    monitor_interval: The interval in seconds that is used by the elastic_agent
                      as a period of monitoring workers.
    start_method: The method is used by the elastic agent to start the
                  workers (spawn, fork, forkserver).
    log_dir: base log directory where log files are written. If not set,
             one is created in a tmp dir but NOT removed on exit.
    redirects: configuration to redirect stdout/stderr to log files.
               Pass a single ``Std`` enum to redirect all workers,
               or a mapping keyed by local_rank to selectively redirect.
    tee: configuration to "tee" stdout/stderr to console + log file.
    metrics_cfg: configuration to initialize metrics.
    """

    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    run_id: str = ""
    role: str = "default_role"
    rdzv_endpoint: str = ""
    rdzv_backend: str = "etcd"
    rdzv_configs: Dict[str, Any] = field(default_factory=dict)
    rdzv_timeout: int = 900
    max_restarts: int = 3
    monitor_interval: float = 30
    start_method: str = "spawn"
    log_dir: Optional[str] = None
    redirects: Union[Std, Dict[int, Std]] = Std.NONE
    tee: Union[Std, Dict[int, Std]] = Std.NONE
    metrics_cfg: Dict[str, str] = field(default_factory=dict)
    project_pactum: bool = False
    max_pipe_parallel_size: int = 1

    def __post_init__(self):
        self.rdzv_configs["timeout"] = self.rdzv_timeout


class elastic_launch:
    """
    Launches an torchelastic agent on the container that invoked the entrypoint.
        1. Pass the ``entrypoint`` arguments as non ``kwargs`` (e.g. no named parameters)/
           ``entrypoint`` can be a function or a command.
        2. The return value is a map of each worker's output mapped
           by their respective global rank.
    Usage
    ::
    def worker_fn(foo):
        # ...
    def main():
        # entrypoint is a function.
        outputs = elastic_launch(ProjectPactumLaunchConfig, worker_fn)(foo)
        # return rank 0's output
        return outputs[0]
        # entrypoint is a command and ``script.py`` is the python module.
        ouptuts = elestic_launch(ProjectPactumLaunchConfig, "script.py")(args)
        ouptuts = elestic_launch(ProjectPactumLaunchConfig, "python")("script.py")
    """

    def __init__(
        self,
        config: ProjectPactumLaunchConfig,
        entrypoint: Union[Callable, str, None],
    ):
        self._config = config
        self._entrypoint = entrypoint

    def __call__(self, *args, **kwargs):
        return launch_agent(self._config, self._entrypoint, list(args))


def _construct_event(config: ProjectPactumLaunchConfig) -> events.Event:
    metadata = {
        "rdzv_backend": config.rdzv_backend,
        "run_id": config.run_id,
        "role": config.role,
    }
    return events.Event(
        name="project_pactum.run.launch_agent",
        source=events.EventSource.AGENT,
        metadata=cast(Dict[str, events.EventMetadataValue], metadata),
    )


def _get_entrypoint_name(
    entrypoint: Union[Callable, str, None], args: List[Any]
) -> str:
    """Retrive entrypoint name with the rule:
    1. If entrypoint is a function, use ``entrypont.__qualname__``.
    2. If entrypoint is a string, check its value:
        2.1 if entrypoint equals to ``sys.executable`` (like "python"), use the first element from ``args``
            which does not start with hifen letter (for example, "-u" will be skipped).
        2.2 otherwise, use ``entrypoint`` value.
    3. Otherwise, return empty string.
    """
    if isinstance(entrypoint, Callable):  # type: ignore[arg-type]
        return entrypoint.__name__  # type: ignore[union-attr]
    elif isinstance(entrypoint, str):
        if entrypoint == sys.executable:
            return next((arg for arg in args if arg[0] != "-"), "")
        else:
            return entrypoint
    else:
        return ""


def _get_addr_and_port(
    rdzv_parameters: RendezvousParameters,
) -> Tuple[Optional[str], Optional[int]]:
    if rdzv_parameters.backend != "static":
        return (None, None)
    endpoint = rdzv_parameters.endpoint
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError(
            "Endpoint is missing in endpoint. Try to add --master_addr and --master_port"
        )
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, default_port=-1)
    if master_port == -1:
        raise ValueError(
            f"port is missing in endpoint: {endpoint}. Try to specify --master_port"
        )
    return (master_addr, master_port)

def config_from_args(args) -> Tuple[ProjectPactumLaunchConfig, Union[Callable, str], List[str]]:
    import os
    from torch.distributed.run import parse_min_max_nnodes, determine_local_world_size, get_rdzv_endpoint
    from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config

    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts >= 0

    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
        omp_num_threads = 1
        print(
            f"*****************************************\n"
            f"Setting OMP_NUM_THREADS environment variable for each process to be "
            f"{omp_num_threads} in default, to avoid your system being overloaded, "
            f"please further tune the variable for optimal performance in "
            f"your application as needed. \n"
            f"*****************************************"
        )
        # This env variable will be passed down to the subprocesses
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)

    if args.rdzv_backend == "static":
        rdzv_configs["rank"] = args.node_rank

    rdzv_endpoint = get_rdzv_endpoint(args)

    config = ProjectPactumLaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        run_id=args.rdzv_id,
        role=args.role,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_backend=args.rdzv_backend,
        rdzv_configs=rdzv_configs,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
        start_method=args.start_method,
        redirects=Std.from_str(args.redirects),
        tee=Std.from_str(args.tee),
        log_dir=args.log_dir,
        project_pactum=args.project_pactum,
        max_pipe_parallel_size=args.max_pipe_parallel_size,
    )

    with_python = not args.no_python
    cmd: Union[Callable, str]
    cmd_args = []
    if args.run_path:
        cmd = run_script_path
        cmd_args.append(args.training_script)
    else:
        if with_python:
            cmd = sys.executable
            cmd_args.append("-u")
            if args.module:
                cmd_args.append("-m")
            cmd_args.append(args.training_script)
        else:
            if not args.use_env:
                raise ValueError(
                    "When using the '--no_python' flag,"
                    " you must also set the '--use_env' flag."
                )
            if args.module:
                raise ValueError(
                    "Don't use both the '--no_python' flag"
                    " and the '--module' flag at the same time."
                )
            cmd = args.training_script
    if not args.use_env:
        log.warning(
            "--use_env is deprecated and will be removed in future releases.\n"
            " Please read local_rank from `os.environ('LOCAL_RANK')` instead."
        )
        cmd_args.append(f"--local_rank={macros.local_rank}")
    cmd_args.extend(args.training_script_args)

    return config, cmd, cmd_args

# pyre-fixme[56]: Pyre was not able to infer the type of the decorator
# torch.distributed.elastic.multiprocessing.errors.record.
@record
def launch_agent(
    config: ProjectPactumLaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
) -> Dict[int, Any]:
    if not config.run_id:
        run_id = str(uuid.uuid4().int)
        logger.warning(f"config has no run_id, generate a new one: {run_id}")
        config.run_id = run_id

    entrypoint_name = _get_entrypoint_name(entrypoint, args)

    logger.info(
        f"Starting elastic_operator with launch configs:\n"
        f"  entrypoint             : {entrypoint_name}\n"
        f"  min_nodes              : {config.min_nodes}\n"
        f"  max_nodes              : {config.max_nodes}\n"
        f"  nproc_per_node         : {config.nproc_per_node}\n"
        f"  run_id                 : {config.run_id}\n"
        f"  rdzv_backend           : {config.rdzv_backend}\n"
        f"  rdzv_endpoint          : {config.rdzv_endpoint}\n"
        f"  rdzv_configs           : {config.rdzv_configs}\n"
        f"  max_restarts           : {config.max_restarts}\n"
        f"  monitor_interval       : {config.monitor_interval}\n"
        f"  log_dir                : {config.log_dir}\n"
        f"  metrics_cfg            : {config.metrics_cfg}\n"
        f"  max_pipe_parallel_size : {config.max_pipe_parallel_size}\n"
    )

    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        run_id=config.run_id,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
        **config.rdzv_configs,
    )

    agent = None
    rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_parameters)
    master_addr, master_port = _get_addr_and_port(rdzv_parameters)
    try:

        if config.project_pactum:
            from project_pactum.agent import ProjectPactumAgent, ProjectPactumWorkerSpec

            assert config.rdzv_backend == 'etcd'

            spec = ProjectPactumWorkerSpec(
                role=config.role,
                local_world_size=config.nproc_per_node,
                entrypoint=entrypoint,
                args=tuple(args),
                rdzv_handler=rdzv_handler,
                max_restarts=config.max_restarts,
                monitor_interval=config.monitor_interval,
                redirects=config.redirects,
                tee=config.tee,
                master_addr=master_addr,
                master_port=master_port,
                max_pipe_parallel_size=config.max_pipe_parallel_size,
            )

            cfg = metrics.MetricsConfig(config.metrics_cfg) if config.metrics_cfg else None
            metrics.initialize_metrics(cfg)

            extra_env = {
                'PROJECT_PACTUM_ENABLED': str(1),
                'PROJECT_PACTUM_ETCD_ENDPOINT': config.rdzv_endpoint,
                'PROJECT_PACTUM_RUN_ID': config.run_id,
                'PROJECT_PACTUM_MAX_PIPE_PARALLEL_SIZE': config.max_pipe_parallel_size,
            }
            agent = ProjectPactumAgent(
                spec=spec, start_method=config.start_method, log_dir=config.log_dir,
                extra_env=extra_env,
            )
        else:
            from torch.distributed.elastic.agent.server.api import WorkerSpec
            from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent

            spec = WorkerSpec(
                role=config.role,
                local_world_size=config.nproc_per_node,
                entrypoint=entrypoint,
                args=tuple(args),
                rdzv_handler=rdzv_handler,
                max_restarts=config.max_restarts,
                monitor_interval=config.monitor_interval,
                redirects=config.redirects,
                tee=config.tee,
                master_addr=master_addr,
                master_port=master_port,
            )

            cfg = metrics.MetricsConfig(config.metrics_cfg) if config.metrics_cfg else None
            metrics.initialize_metrics(cfg)

            agent = LocalElasticAgent(
                spec=spec, start_method=config.start_method, log_dir=config.log_dir
            )

        result = agent.run()
        events.record(agent.get_agent_status_event(WorkerState.SUCCEEDED))
        if result.is_failed():
            # ChildFailedError is treated specially by @record
            # if the error files for the failed children exist
            # @record will copy the first error (root cause)
            # to the error file of the launcher process.
            raise ChildFailedError(
                name=entrypoint_name,
                failures=result.failures,
            )
        else:
            return result.return_values
    except ChildFailedError:
        raise
    except Exception:
        if agent:
            events.record(agent.get_agent_status_event(WorkerState.FAILED))
        else:
            events.record(_construct_event(config))
        raise
    finally:
        rdzv_handler.shutdown()
