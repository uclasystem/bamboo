import dataclasses
import datetime
import enum
import heapq
import logging
import math
import random
import statistics
import typing

logger = logging.getLogger(__name__)

class EventKind(enum.IntEnum):
    SPOT_INSTANCE_ADD = 1
    SPOT_INSTANCE_REMOVE = 2
    SPOT_INSTANCE_GENERATE = 3
    SPOT_INSTANCE_READY = 4
    GLOBAL_RENDEZVOUS_TIMEOUT = 5
    LOCAL_RENDEZVOUS_TIMEOUT = 6
    TRAINING_STEP_COMPLETE = 7

@dataclasses.dataclass(order=True)
class Event:
    delta: int
    kind: EventKind
    data: typing.Dict = dataclasses.field(compare=False)

class SystemStatus(enum.Enum):
    STOPPED = enum.auto()
    RENDEZVOUS = enum.auto()
    RUNNING = enum.auto()

class NodeStatus(enum.Enum):
    CREATING = enum.auto()
    READY = enum.auto()
    RUNNING = enum.auto()

@dataclasses.dataclass
class Result:
    removal_probability: float
    preemption_mean: float
    preemption_median: float
    preemption_stdev: float
    lifetime_mean: float
    lifetime_median: float
    lifetime_stdev: float
    num_preemptions: int
    num_fatal_failures: int
    num_steps_complete: int
    average_instances: float
    average_performance: float
    average_cost: float
    average_value: float

class SpotInstance:
    def __init__(self, name, start):
        self.name = name
        self.start = start
        self.status = NodeStatus.CREATING
        self.global_id = None
        self.active_coordinates = []
        self.previous_coordinates = []

    def is_creating(self):
        return self.status == NodeStatus.CREATING

    def set_ready(self):
        self.status = NodeStatus.READY

    def is_ready(self):
        return self.status == NodeStatus.READY

    def set_running(self):
        self.status = NodeStatus.RUNNING

    def is_running(self):
        return self.status == NodeStatus.RUNNING

    def uptime(self, end):
        return end - self.start

    def __str__(self):
        return self.name

class Simulator:

    def __init__(self,
                 seed=None,
                 start_hour=None,
                 generate_addition_probabilities=False,
                 removal_probability=None,
                 generate_graphs=False):
        self.generate_graphs = generate_graphs

        self.seed = seed
        if self.seed is not None:
            self.r = random.Random(self.seed)
            logger.info(f'Using seed: {self.seed}')
        else:
            self.r = random.Random()
        self.generate_addition_probabilities = generate_addition_probabilities
        self.removal_probability = removal_probability

        self.hour = datetime.timedelta(hours=1)
        self.second = datetime.timedelta(seconds=1)
        self.millisecond = datetime.timedelta(milliseconds=1)
        self.milliseconds_per_second = self.second / self.millisecond
        self.milliseconds_per_hour = self.hour / self.millisecond

        self.spot_instance_name_format = 'node{id}'
        self.spot_instance_next_id = 1
        self.spot_instance_desired_capacity = 64
        if not generate_addition_probabilities:
            self.spot_instance_addition_probability = {
                0: 0.05,
                1: 0.05,
                2: 0.05,
                3: 0.50,
                4: 0.50,
                5: 0.50,
                6: 0.05,
                7: 0.05,
                8: 0.05,
                9: 0.05,
                10: 0.05,
                11: 0.05,
                12: 0.05,
                13: 0.05,
                14: 0.05,
                15: 0.00,
                16: 0.00,
                17: 0.00,
                18: 0.00,
                19: 0.00,
                20: 0.00,
                21: 0.00,
                22: 0.00,
                23: 0.05,
            }
        else:
            self.spot_instance_addition_probability = self.generate_probabilities()
        if removal_probability is None:
            self.spot_instance_removal_probability = {
                0: 0.05,
                1: 0.05,
                2: 0.05,
                3: 0.01,
                4: 0.01,
                5: 0.01,
                6: 0.05,
                7: 0.05,
                8: 0.05,
                9: 0.05,
                10: 0.05,
                11: 0.05,
                12: 0.05,
                13: 0.05,
                14: 0.05,
                15: 0.25,
                16: 0.25,
                17: 0.25,
                18: 0.25,
                19: 0.25,
                20: 0.25,
                21: 0.25,
                22: 0.25,
                23: 0.05,
            }
        else:
            self.spot_instance_removal_probability = {}
            for hour in range(24):
                self.spot_instance_removal_probability[hour] = removal_probability
        self.start_hour = start_hour
        self.spot_instance_creation_time = 45_000 # milliseconds
        self.global_rendezvous_timeout_delta = 30_000 # milliseconds
        self.local_rendezvous_timeout_delta = 1_000 # milliseconds
        self.time_for_single_pipeline = 10_000 # milliseconds

        self.equal_performance_factor = 1.5
        self.num_stages_minimum = 6
        self.on_demand_num_stages = self.num_stages_minimum / self.equal_performance_factor
        assert self.on_demand_num_stages % 1.0 == 0.0
        self.on_demand_num_stages = int(self.on_demand_num_stages)
        self.on_demand_price_factor = 0.3

        self.spot_instances = {}
        self.rendezvous_version = 0
        self.rendezvous = []
        self.num_workers_waiting = 0
        self.num_pipelines = 0
        self.num_stages = 0

        self.num_steps_complete = 0

        self.num_fatal_failures = 0
        self.num_spot_instance_removals = 0
        self.spot_instance_removal_times = []
        self.spot_instance_lifetimes = []

        self.previous_step_complete_delta = 0
        self.tokens_per_iteration = 16_384
        self.iterations_per_job = 183_106

        self.performance_xs = []
        self.performance_ys = []

        self.cost_dollars_per_hour_per_instance = 0.90
        self.cost_xs = []
        self.cost_ys = []

        self.value_xs = []
        self.value_ys = []

        self.events = []
        heapq.heapify(self.events)

        self.status = SystemStatus.STOPPED

    def generate_probabilities(self):
        probability = {}
        for hour in range(24):
            probability[hour] = self.r.random()
        return probability

    def simulate_step_delta(self):
        # TODO: I need help with this, it shouldn't be constant
        self.step_delta = int(self.time_for_single_pipeline / self.num_pipelines)

        num_workers_overloaded = self.get_num_workers_overloaded()
        if num_workers_overloaded > 1:
            self.step_delta = int(self.step_delta * 1.5)

    def info(self, delta, message):
        logger.info(f'[{delta/1000.0:.3f}] {message}')
 
    def get_spot_instance_next_name(self):
        name = self.spot_instance_name_format.format(
            id=self.spot_instance_next_id
        )
        self.spot_instance_next_id += 1
        return name

    def generate_spot_instance_probability_delta(self,
                                                 current_time,
                                                 current_delta,
                                                 probability):
        hour = current_time.hour
        p = probability[hour]
        if p > self.r.random():
            local_delta = datetime.timedelta(seconds=self.r.randrange(0, 3600))
            delta = current_delta + local_delta
            return delta
        return None

    def generate_spot_instance_addition_delta(self,
                                              current_time,
                                              current_delta):
        return self.generate_spot_instance_probability_delta(
            current_time,
            current_delta,
            self.spot_instance_addition_probability
        )

    def generate_spot_instance_removal_delta(self,
                                             current_time,
                                             current_delta):
        return self.generate_spot_instance_probability_delta(
            current_time,
            current_delta,
            self.spot_instance_removal_probability
        )

    def create_event(self, delta, kind, data):
        if isinstance(delta, datetime.timedelta):
            event = Event(delta // self.millisecond, kind, data)
        else:
            assert type(delta) == int
            event = Event(delta, kind, data)
        heapq.heappush(self.events, event)
        return event

    def create_spot_instance_generate_event(self, delta):
        return self.create_event(delta, EventKind.SPOT_INSTANCE_GENERATE, {})

    def create_spot_instance_add_event(self, delta):
        name = self.get_spot_instance_next_name()
        return self.create_event(delta, EventKind.SPOT_INSTANCE_ADD, {
            'name': name,
        })

    def create_spot_instance_remove_event(self, delta, name):
        return self.create_event(delta, EventKind.SPOT_INSTANCE_REMOVE, {
            'name': name,
        })

    def create_spot_instance_ready_event(self, delta, name):
        return self.create_event(delta, EventKind.SPOT_INSTANCE_READY, {
            'name': name,
        })

    def create_global_rendezvous_timeout_event(self, delta):
        return self.create_event(
            delta + self.global_rendezvous_timeout_delta,
            EventKind.GLOBAL_RENDEZVOUS_TIMEOUT,
            {}
        )

    def create_local_rendezvous_timeout_event(self, delta):
        return self.create_event(
            delta + self.local_rendezvous_timeout_delta,
            EventKind.LOCAL_RENDEZVOUS_TIMEOUT,
            {}
        )

    def create_training_step_complete_event(self, delta, rendezvous_version):
        return self.create_event(
            delta + self.step_delta,
            EventKind.TRAINING_STEP_COMPLETE,
            {'rendezvous_version': rendezvous_version}
        )

    def generate_spot_instance_initial_events(self, start):
        # Generate the initial instances
        spot_instance_initial_probability = self.spot_instance_addition_probability[start.hour]
        delta = 0
        for i in range(self.spot_instance_desired_capacity):
            if spot_instance_initial_probability > self.r.random():
                event = self.create_spot_instance_add_event(delta)
        self.create_spot_instance_generate_event(delta)

    # def generate_spot_instance_events(self, start, duration): # TODO
    def generate_spot_instance_events(self, start, delta):
        current_delta = delta * self.millisecond
        self.create_spot_instance_generate_event(current_delta + self.hour)

        current_instances = {}
        for name, instance in self.spot_instances.items():
            current_instances[name] = current_delta

        # The remaining code generates add and remove events for the next hour
        current_time = start + current_delta

        # Run removal for currently running nodes
        removed_instances = []
        for name in current_instances.keys():
            delta = self.generate_spot_instance_removal_delta(current_time,
                                                              current_delta)
            if delta is None:
                continue
            removed_instances.append((delta, name))
        heapq.heapify(removed_instances)

        # Run additions for the maximum number of instances you ever need
        # this hour
        possible_added_deltas = []
        requested_capacity = self.spot_instance_desired_capacity - len(current_instances) + len(removed_instances)
        for _ in range(requested_capacity):
            delta = self.generate_spot_instance_addition_delta(current_time, current_delta)
            if delta is None:
                continue
            possible_added_deltas.append(delta)
        heapq.heapify(possible_added_deltas)

        # Check if we have to throw out any additions because we're already
        # at capacity
        while len(removed_instances) > 0 or len(possible_added_deltas) > 0:
            # The next event is a removal, so just do it
            if len(removed_instances) > 0 and (len(possible_added_deltas) == 0 or removed_instances[0][0] < possible_added_deltas[0]):
                delta, name = heapq.heappop(removed_instances)

                event = self.create_spot_instance_remove_event(delta, name)

                del current_instances[name]
            # The next event is an addition, only do it if we're under
            # desired capacity
            else:
                delta = heapq.heappop(possible_added_deltas)

                # Skip this addition
                if len(current_instances) == self.spot_instance_desired_capacity:
                    continue

                event = self.create_spot_instance_add_event(delta)

                name = event.data['name']
                current_instances[name] = delta

                # Check if we also attempt to remove this new instance in
                # this hour
                delta = self.generate_spot_instance_removal_delta(
                    current_time,
                    current_delta
                )
                if delta is None or delta < current_instances[name]:
                    continue
                heapq.heappush(removed_instances, (delta, name))

    def append_cost(self, delta):
        self.cost_xs.append(delta / self.milliseconds_per_hour)
        self.cost_ys.append(
            self.cost_dollars_per_hour_per_instance * len(self.spot_instances)
        )
        self.append_value(delta)

    def append_value(self, delta):
        if len(self.performance_ys) == 0 or len(self.cost_ys) == 0:
            return
        if self.cost_ys[-1] == 0.0:
            return

        self.value_xs.append(delta / self.milliseconds_per_hour)
        self.value_ys.append(
            self.performance_ys[-1] / self.cost_ys[-1]
        )

    def simulate_spot_instance_add(self, delta, data):
        name = data['name']
        self.spot_instances[name] = SpotInstance(name, delta)
        self.create_spot_instance_ready_event(
            delta + self.spot_instance_creation_time,
            name,
        )

        self.append_cost(delta)

    def simulate_fatal_failure(self, delta, name):
        self.info(
            delta,
            f'{name} caused a fatal failure, starting global rendezvous'
        )
        self.num_fatal_failures += 1
        self.simulate_rendezvous_start(delta, True)

    def simulate_spot_instance_remove(self, delta, data):
        name = data['name']
        instance = self.spot_instances[name]

        self.num_spot_instance_removals += 1
        self.spot_instance_lifetimes.append(delta - instance.start)
        self.spot_instance_removal_times.append(delta)
        del self.spot_instances[name]

        self.append_cost(delta)

        if instance.is_running():
            # This is a fatal failure
            if len(instance.active_coordinates) > 1:
                self.simulate_fatal_failure(delta, name)
                return

            coordinates = instance.active_coordinates[0]
            # Find which node has my redundant coordinates
            search = (coordinates[0], coordinates[1] - 1)
            if search[1] == -1:
                search = (search[0], self.num_stages - 1)
            for n, i in self.spot_instances.items():
                found = False
                for c in i.active_coordinates:
                    if c == search:
                        # This node recovered previously, so it doesn't have
                        # the redundant coordinates
                        if len(i.active_coordinates) > 1:
                            self.simulate_fatal_failure(delta, name)
                            return
                        i.active_coordinates.append(coordinates)
                        found = True
                        break
                if found:
                    break
        
        # Re-simulate the step delta now that we lost a node
        if self.status == SystemStatus.RUNNING:
            self.simulate_step_delta()

    def simulate_rendezvous_start(self, delta, is_global):
        self.status = SystemStatus.RENDEZVOUS
        self.simulate_rendezvous_restart(delta)
        if is_global:
            self.create_global_rendezvous_timeout_event(delta)
        else:
            self.create_local_rendezvous_timeout_event(delta)

    def simulate_rendezvous_restart(self, delta):
        assert self.status == SystemStatus.RENDEZVOUS
        self.rendezvous = []
        for name, instance, in self.spot_instances.items():
            if instance.is_ready() or instance.is_running():
                self.rendezvous.append(name)

    def simulate_rendezvous_timeout(self, delta, is_global):
        for i, name in enumerate(self.rendezvous):
            if name not in self.spot_instances:
                self.info(
                    delta,
                    f'{name} terminated during redezvous, restarting'
                )
                self.simulate_rendezvous_restart(delta)
                if is_global:
                   self.create_global_rendezvous_timeout_event(delta)
                else:
                   self.create_local_rendezvous_timeout_event(delta)
                return
            instance = self.spot_instances[name]
            instance.global_id = i
        self.simulate_assign_coordinates()
        self.rendezvous_version += 1
        self.info(
            delta,
            f'{len(self.rendezvous)} nodes completed rendezvous version '
            f'{self.rendezvous_version}, '
            f'{self.num_pipelines}x{self.num_stages} configuration'
        )
        self.rendezvous = []
        if self.num_pipelines != 0:
            self.status = SystemStatus.RUNNING
            self.simulate_step_delta()
            self.create_training_step_complete_event(delta,
                                                     self.rendezvous_version)
        else:
            self.status = SystemStatus.STOPPED

    def simulate_spot_instance_ready(self, delta, data):
        name = data['name']

        # This node has already been removed
        if name not in self.spot_instances:
            return

        instance = self.spot_instances[name]
        instance.set_ready()

        if self.status == SystemStatus.STOPPED:
            self.info(delta, f'{name} starting global rendezvous')
            self.simulate_rendezvous_start(delta, True)
        elif self.status == SystemStatus.RENDEZVOUS:
            self.rendezvous.append(name)
        elif self.status == SystemStatus.RUNNING:
            self.num_workers_waiting += 1

    def simulate_global_rendezvous_timeout(self, delta, data):
        self.simulate_rendezvous_timeout(delta, True)

    def simulate_local_rendezvous_timeout(self, delta, data):
        self.simulate_rendezvous_timeout(delta, False)

    def simulate_assign_coordinates(self):
        if len(self.rendezvous) < self.num_stages_minimum:
            num_pipelines = 0
            num_stages = 0
        else:
            num_stages = self.num_stages_minimum
            num_pipelines = len(self.rendezvous) // num_stages
        num_workers_waiting = 0

        previous_num_pipelines = self.num_pipelines
        previous_num_stages = self.num_stages

        required_coordinates = []
        for i in range(num_pipelines):
            for j in range(num_stages):
                required_coordinates.append((i, j))

        for name in self.rendezvous:
            instance = self.spot_instances[name]
            assert instance.is_ready() or instance.is_running()
            instance.previous_coordinates = instance.active_coordinates
            try:
                coordinates = required_coordinates.pop(0)
                instance.active_coordinates = [coordinates]
                instance.set_running()
            except IndexError:
                instance.active_coordinates = []
                instance.set_ready()
                num_workers_waiting += 1

        self.num_pipelines = num_pipelines
        self.num_stages = num_stages
        self.num_workers_waiting = num_workers_waiting

    def get_num_workers_overloaded(self):
        num_workers_overloaded = 0
        for name, instance in self.spot_instances.items():
            if not instance.is_running():
                continue
            elif len(instance.active_coordinates) > 1:
                assert len(instance.active_coordinates) == 2
                num_workers_overloaded += 1
        return num_workers_overloaded

    def simulate_should_reconfigure(self):
        num_workers_overloaded = self.get_num_workers_overloaded()
        num_workers_waiting = self.num_workers_waiting

        # If we can easily re-balance, do it
        if num_workers_overloaded > 0 and num_workers_waiting >= num_workers_overloaded:
            return True

        num_original_workers = self.num_pipelines * self.num_stages
        num_active_workers = num_original_workers - num_workers_overloaded

        # If we're above a 5% chance of failure, re-configure/re-balance
        # I cannot remove an overloaded node, or the node that now has no
        # redundancy
        if num_workers_overloaded > 0 and (2 * num_workers_overloaded / num_active_workers) > 0.05:
            return True

        # If we can add another pipeline, do it
        potential_num_pipelines = (num_active_workers + num_workers_waiting) // self.num_stages_minimum
        if potential_num_pipelines > self.num_pipelines:
            return True

        return False

    def simulate_training_step_complete(self, delta, data):
        rendezvous_version = data['rendezvous_version']
        if rendezvous_version != self.rendezvous_version:
            return

        self.num_steps_complete += 1
        # Calculate performance
        time_to_step_complete = delta - self.previous_step_complete_delta
        tokens_per_second = self.tokens_per_iteration / (time_to_step_complete / self.milliseconds_per_second)
        self.performance_xs.append(delta / self.milliseconds_per_hour)
        self.performance_ys.append(tokens_per_second)
        self.previous_step_complete_delta = delta

        if self.num_steps_complete % 100 == 0:
            self.info(delta, f'{self.num_steps_complete} steps complete')
        if self.simulate_should_reconfigure():
            self.info(
                delta,
                f'reconfiguration after step {self.num_steps_complete}'
            )
            self.simulate_rendezvous_start(delta, False)
        else:
            self.create_training_step_complete_event(
                delta,
                self.rendezvous_version
            )

    def calculate_average(self, xs, ys, duration):
        previous_x = 0.0
        previous_y = 0.0
        total = 0.0
        for x, y in zip(xs, ys):
            total += (x - previous_x) * previous_y
            previous_x = x
            previous_y = y
        total += (duration - previous_x) * previous_y
        return total / duration
        
    def simulate(self, duration=None):
        start = datetime.datetime.now(datetime.timezone.utc)
        start = start.replace(minute=0, second=0, microsecond=0)
        if self.start_hour is not None:
            start = start.replace(hour=self.start_hour)
        #duration = datetime.timedelta(hours=2) // self.millisecond
        #duration = datetime.timedelta(days=2) // self.millisecond
        # duration_milliseconds = duration // self.millisecond
        # end = start + duration

        logger.info(f'Starting at {start}')

        logger.info(f'Generating spot instance events...')
        self.generate_spot_instance_initial_events(start)
        #logger.info(f'Generating spot instance events...')
        #self.generate_spot_instance_events(start, duration)

        instances_xs = []
        instances_ys = []
        while len(self.events) > 0:
            event = heapq.heappop(self.events)

            kind = event.kind
            delta = event.delta
            data = event.data
            # print('HELLO?', delta, type(delta))

            if duration is not None and delta > duration:
                delta = duration
                break

            if kind == EventKind.SPOT_INSTANCE_ADD:
                self.simulate_spot_instance_add(delta, data)
            elif kind == EventKind.SPOT_INSTANCE_REMOVE:
                self.simulate_spot_instance_remove(delta, data)
            elif kind == EventKind.SPOT_INSTANCE_GENERATE:
                self.generate_spot_instance_events(start, delta)
            elif kind == EventKind.SPOT_INSTANCE_READY:
                self.simulate_spot_instance_ready(delta, data)
            elif kind == EventKind.GLOBAL_RENDEZVOUS_TIMEOUT:
                self.simulate_global_rendezvous_timeout(delta, data)
            elif kind == EventKind.LOCAL_RENDEZVOUS_TIMEOUT:
                self.simulate_local_rendezvous_timeout(delta, data)
            elif kind == EventKind.TRAINING_STEP_COMPLETE:
                self.simulate_training_step_complete(delta, data)
            else:
                raise ValueError(f'Unknown kind: {kind}')

            # We're done our training
            if duration is None and self.num_steps_complete == self.iterations_per_job:
                break

            # We still need to process more events for this delta
            next_event = self.events[0] if len(self.events) > 0 else None
            next_delta = next_event.delta if next_event else None
            if delta == next_delta:
                continue

            instances_xs.append(delta/self.milliseconds_per_hour)
            instances_ys.append(len(self.spot_instances))

        duration_hours = math.ceil(delta / self.milliseconds_per_hour)

        # Complete the remaining
        for name, instance in self.spot_instances.items():
            self.spot_instance_lifetimes.append(
                delta - instance.start
            )

        spot_instance_between_removal_times = []
        previous_removal_time = self.spot_instance_removal_times[0] \
            if len(self.spot_instance_removal_times) > 0 else None
        for removal_time in self.spot_instance_removal_times[1:]:
            spot_instance_between_removal_times.append(
                removal_time - previous_removal_time
            )
            previous_removal_time = removal_time

        result = Result(
            removal_probability = self.removal_probability,
            preemption_mean = statistics.mean(spot_instance_between_removal_times) / self.milliseconds_per_hour,
            preemption_median = statistics.mean(spot_instance_between_removal_times) / self.milliseconds_per_hour,
            preemption_stdev = statistics.stdev(spot_instance_between_removal_times) / self.milliseconds_per_hour if len(spot_instance_between_removal_times) > 1 else 0,
            lifetime_mean = statistics.mean(self.spot_instance_lifetimes) / self.milliseconds_per_hour,
            lifetime_median = statistics.median(self.spot_instance_lifetimes) / self.milliseconds_per_hour,
            lifetime_stdev = statistics.stdev(self.spot_instance_lifetimes) / self.milliseconds_per_hour,
            num_preemptions = self.num_spot_instance_removals,
            num_fatal_failures = self.num_fatal_failures,
            num_steps_complete = self.num_steps_complete,
            average_instances = self.calculate_average(instances_xs, instances_ys, duration_hours),
            average_performance = self.calculate_average(self.performance_xs, self.performance_ys, duration_hours),
            average_cost = self.calculate_average(self.cost_xs, self.cost_ys, duration_hours),
            average_value = self.calculate_average(self.value_xs, self.value_ys, duration_hours),
        )

        if self.generate_graphs:
        # while True:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import StrMethodFormatter
            # sizes: xx-small, x-small, small, medium, large, x-large, xx-large
            params = {
                'font.family': 'Inter',
                'legend.fontsize': 'medium',
                'axes.labelsize': 'medium',
                'axes.titlesize': 'medium',
                'xtick.labelsize': 'medium',
                'ytick.labelsize': 'medium',
            }
            plt.rcParams.update(params)

            plt.plot(instances_xs, instances_ys)
            plt.xlim(0, duration_hours)
            plt.xlabel('Time (hours)')
            plt.ylabel('# Instances')
            plt.xticks(range(0, duration_hours + 1, 12))
            plt.hlines(result.average_instances, 0, duration_hours, color='tab:blue', linestyles='dashed')
            
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.axis([0, duration_hours, 0, 64])

            pdf_suffix = f'-seed-{self.seed}-start-hour-{self.start_hour}-generate-addition-probabilities-{self.generate_addition_probabilities}-removal-probability-{self.removal_probability}.pdf'

            plt.savefig(
                f'instances{pdf_suffix}',
                bbox_inches='tight',
                pad_inches=0
            )
            plt.show()

            plt.clf()

            # Calculate on demand performance
            on_demand_num_pipelines = self.spot_instance_desired_capacity // self.num_stages_minimum
            on_demand_num_instances = on_demand_num_pipelines * self.on_demand_num_stages
            on_demand_time_to_step_complete = self.time_for_single_pipeline / on_demand_num_pipelines
            on_demand_tokens_per_second = self.tokens_per_iteration / (on_demand_time_to_step_complete / self.milliseconds_per_second)
            plt.hlines(on_demand_tokens_per_second, 0, duration_hours, color='red')

            plt.hlines(result.average_performance, 0, duration_hours, color='tab:blue', linestyles='dashed')
            plt.plot(self.performance_xs, self.performance_ys)
            plt.xlim(0, duration_hours)
            plt.ylim(0)
            plt.xlabel('Time (hours)')
            plt.ylabel('Performance (tokens per second)')
            plt.xticks(range(0, duration_hours + 1, 12))
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.savefig(
                f'performance{pdf_suffix}',
                bbox_inches='tight',
                pad_inches=0
            )
            plt.show()

            plt.clf()

            on_demand_cost_per_hour = self.cost_dollars_per_hour_per_instance / self.on_demand_price_factor * on_demand_num_instances
            plt.hlines(result.average_cost, 0, duration_hours, color='tab:blue', linestyles='dashed')
            plt.hlines(on_demand_cost_per_hour, 0, duration_hours, color='red')

            plt.plot(self.cost_xs, self.cost_ys)
            plt.xlim(0, duration_hours)
            plt.ylim(0)
            plt.xlabel('Time (hours)')
            plt.ylabel('Cost ($ per hour)')
            plt.xticks(range(0, duration_hours + 1, 12))
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.savefig(
                f'cost{pdf_suffix}',
                bbox_inches='tight',
                pad_inches=0
            )
            plt.show()

            plt.hlines(
                on_demand_tokens_per_second / on_demand_cost_per_hour,
                0, duration_hours, color='red'
            )
            plt.hlines(result.average_value, 0, duration_hours, color='tab:blue', linestyles='dashed')
            plt.plot(self.value_xs, self.value_ys)
            plt.xlim(0, duration_hours)
            plt.ylim(0)
            plt.xlabel('Time (hours)')
            plt.ylabel('Value (performance per cost)')
            plt.xticks(range(0, duration_hours + 1, 12))
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            plt.savefig(
                f'value{pdf_suffix}',
                bbox_inches='tight',
                pad_inches=0
            )
            plt.show()


        # print('Preemptions')
        # print('  - Mean:', result.preemption_mean, 'hours')
        # print('  - Median:', result.preemption_median, 'hours')
        # print('  - Stdev:', result.preemption_stdev, 'hours')
        # print('Lifetimes')
        # print('  - Mean:', result.lifetime_mean, 'hours')
        # print('  - Median:', result.lifetime_median, 'hours')
        # print('  - Stdev:', result.lifetime_stdev, 'hours')
        # print('Number of preemptions:', result.num_preemptions)
        # print('Number of fatal failures:', result.num_fatal_failures)
        # print('Number of steps complete:', result.num_steps_complete)

        logger.info(f'Ending after {duration_hours} hours')

        return result
