import dataclasses
import datetime
import enum
import heapq
import logging
import math
import random
import typing

logger = logging.getLogger(__name__)

class EventKind(enum.IntEnum):
    SPOT_INSTANCE_ADD = 1
    SPOT_INSTANCE_REMOVE = 2
    SPOT_INSTANCE_READY = 3
    GLOBAL_RENDEZVOUS_TIMEOUT = 4
    TRAINING_STEP_COMPLETE = 5

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
    GLOBAL_RENDEZVOUS = enum.auto()
    LOCAL_RENDEZVOUS = enum.auto()
    RUNNING = enum.auto()

class SpotInstance:
    def __init__(self, name, start):
        self.name = name
        self.start = start
        self.status = NodeStatus.CREATING
        self.global_id = None

    def set_ready(self):
        self.status = NodeStatus.READY

    def set_global_rendezvous(self):
        self.status = NodeStatus.GLOBAL_RENDEZVOUS

    def set_local_rendezvous(self):
        self.status = NodeStatus.LOCAL_RENDEZVOUS

    def set_running(self):
        self.status = NodeStatus.RUNNING

    def set_global_id(self, global_id):
        self.global_id = global_id

    def uptime(self, end):
        return end - self.start

    def __str__(self):
        return self.name

class Simulator:

    def __init__(self, options):
        self.generate_graphs = options.generate_graphs

        #self.seed = 12345
        #self.r = random.Random(self.seed)
        self.r = random.Random()

        self.millisecond = datetime.timedelta(milliseconds=1)

        self.spot_instance_name_format = 'node{id}'
        self.spot_instance_next_id = 1
        self.spot_instance_desired_capacity = 64
        self.spot_instance_initial_probability = 1.00
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
        self.spot_instance_creation_time = 45_000 # milliseconds
        self.global_rendezvous_time = 30_000 # milliseconds
        self.time_per_step = 30_000 # milliseconds

        self.spot_instances = {}
        self.rendezvous = []

        self.events = []
        heapq.heapify(self.events)

        self.status = SystemStatus.STOPPED
 
    def get_spot_instance_next_name(self):
        name = self.spot_instance_name_format.format(id=self.spot_instance_next_id)
        self.spot_instance_next_id += 1
        return name

    def generate_spot_instance_probability_delta(self, current_time, current_delta, probability):
        hour = current_time.hour
        p = probability[hour]
        if p > self.r.random():
            local_delta = datetime.timedelta(seconds=self.r.randrange(0, 3600))
            delta = current_delta + local_delta
            return delta
        return None

    def generate_spot_instance_addition_delta(self, current_time, current_delta):
        return self.generate_spot_instance_probability_delta(current_time, current_delta,
                                                             self.spot_instance_addition_probability)

    def generate_spot_instance_removal_delta(self, current_time, current_delta):
        return self.generate_spot_instance_probability_delta(current_time, current_delta,
                                                             self.spot_instance_removal_probability)

    def create_event(self, delta, kind, data):
        if isinstance(delta, datetime.timedelta):
            event = Event(delta // self.millisecond, kind, data)
        else:
            event = Event(delta, kind, data)
        heapq.heappush(self.events, event)
        return event

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
        return self.create_event(delta, EventKind.GLOBAL_RENDEZVOUS_TIMEOUT, {})

    def create_training_step_complete_event(self, delta):
        return self.create_event(delta, EventKind.TRAINING_STEP_COMPLETE, {})

    def generate_spot_instance_events(self, start, duration):
        current_delta = datetime.timedelta(seconds=0)
        current_instances = {}

        for i in range(self.spot_instance_desired_capacity):
            if self.spot_instance_initial_probability > self.r.random():
                event = self.create_spot_instance_add_event(current_delta)
                current_instances[event.data['name']] = event.delta

        while duration > current_delta:
            current_time = start + current_delta

            # Run removal for currently running nodes
            removed_instances = []
            for name in current_instances.keys():
                delta = self.generate_spot_instance_removal_delta(current_time, current_delta)
                if delta is None:
                    continue
                removed_instances.append((delta, name))
            heapq.heapify(removed_instances)

            # Run additions for the maximum number of instances you ever need this hour
            possible_added_deltas = []
            requested_capacity = self.spot_instance_desired_capacity - len(current_instances) + len(removed_instances)
            for _ in range(requested_capacity):
                delta = self.generate_spot_instance_addition_delta(current_time, current_delta)
                if delta is None:
                    continue
                possible_added_deltas.append(delta)
            heapq.heapify(possible_added_deltas)

            # Check if we have to throw out any additions because we're already at capacity
            while len(removed_instances) > 0 or len(possible_added_deltas) > 0:
                # The next event is a removal, so just do it
                if len(removed_instances) > 0 and (len(possible_added_deltas) == 0 or removed_instances[0][0] < possible_added_deltas[0]):
                    delta, name = heapq.heappop(removed_instances)

                    event = self.create_spot_instance_remove_event(delta, name)

                    del current_instances[name]
                # The next event is an addition, only do it if we're under desired capacity
                else:
                    delta = heapq.heappop(possible_added_deltas)

                    # Skip this addition
                    if len(current_instances) == self.spot_instance_desired_capacity:
                        continue

                    event = self.create_spot_instance_add_event(delta)

                    name = event.data['name']
                    current_instances[name] = delta

                    # Check if we also attempt to remove this new instance in this hour
                    delta = self.generate_spot_instance_removal_delta(current_time, current_delta)
                    if delta is None or delta < current_instances[name]:
                        continue
                    heapq.heappush(removed_instances, (delta, name))

            current_delta += datetime.timedelta(hours=1)

    def simulate_spot_instance_add(self, delta, data):
        name = data['name']
        self.spot_instances[name] = SpotInstance(name, delta)
        self.create_spot_instance_ready_event(
            delta + self.spot_instance_creation_time,
            name,
        )

    def simulate_spot_instance_remove(self, delta, data):
        name = data['name']
        del self.spot_instances[name]

    def simulate_spot_instance_ready(self, delta, data):
        name = data['name']

        # This node has already been removed
        if name not in self.spot_instances:
            return

        instance = self.spot_instances[name]
        instance.set_ready()

        if self.status == SystemStatus.STOPPED:
            logger.info(f'{name} starting global rendezvous')
            self.create_global_rendezvous_timeout_event(delta + self.global_rendezvous_time)
            self.status = SystemStatus.RENDEZVOUS

        if self.status == SystemStatus.RENDEZVOUS:
            instance.set_global_rendezvous()
            self.rendezvous.append(name)

    def simulate_global_rendezvous_timeout(self, delta, data):
        logger.info(f'{len(self.rendezvous)} nodes joined global rendezvous')
        for i, name in enumerate(self.rendezvous):
            if name not in self.spot_instances:
                continue
            instance = self.spot_instances[name]
            instance.set_global_id(i)
            instance.set_running()
        self.status = SystemStatus.RUNNING
        self.create_training_step_complete_event(delta + self.time_per_step)

    def simulate_training_step_complete(self, delta, data):
        self.create_training_step_complete_event(delta + self.time_per_step)

    def simulate(self):
        start = datetime.datetime.now(datetime.timezone.utc)
        start = start.replace(minute=0, second=0, microsecond=0)
        duration = datetime.timedelta(days=2)
        duration_milliseconds = duration // self.millisecond
        end = start + duration

        logger.info(f'Starting at {start}')

        logger.info(f'Generating spot instance events...')
        self.generate_spot_instance_events(start, duration)

        xs = []
        ys = []
        while len(self.events) > 0:
            event = heapq.heappop(self.events)

            kind = event.kind
            delta = event.delta
            data = event.data

            if delta > duration_milliseconds:
                break

            if kind == EventKind.SPOT_INSTANCE_ADD:
                self.simulate_spot_instance_add(delta, data)
            elif kind == EventKind.SPOT_INSTANCE_REMOVE:
                self.simulate_spot_instance_remove(delta, data)
            elif kind == EventKind.SPOT_INSTANCE_READY:
                self.simulate_spot_instance_ready(delta, data)
            elif kind == EventKind.GLOBAL_RENDEZVOUS_TIMEOUT:
                self.simulate_global_rendezvous_timeout(delta, data)
            elif kind == EventKind.TRAINING_STEP_COMPLETE:
                self.simulate_training_step_complete(delta, data)
            else:
                raise ValueError(f'Unknown kind: {kind}')

            # We still need to process more events for this delta
            next_event = self.events[0] if len(self.events) > 0 else None
            next_delta = next_event.delta if next_event else None
            if delta == next_delta:
                continue

            status = self.status
            #if status == SystemStatus.STOPPED:
            #    for name, instance in instances.items():
            #        print(type(instance))
            #    status = SystemStatus.RENDEZVOUS
            # print(self.status)

            xs.append(delta)
            ys.append(len(self.spot_instances))

        if self.generate_graphs:
            import matplotlib.pyplot as plt
            plt.plot(xs, ys)
            plt.axis([0, duration//self.millisecond, 0, 64])
            plt.show()

        logger.info(f'Ending at {end}')
