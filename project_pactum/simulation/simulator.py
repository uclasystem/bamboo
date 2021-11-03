import datetime
import enum
import heapq
import logging
import math
import random

logger = logging.getLogger(__name__)

class SpotInstance:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

class Simulator:

    def __init__(self):
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
            3: 0.05,
            4: 0.05,
            5: 0.05,
            6: 0.05,
            7: 0.05,
            8: 0.05,
            9: 0.05,
            10: 0.05,
            11: 0.05,
            12: 0.05,
            13: 0.05,
            14: 0.05,
            15: 0.05,
            16: 0.05,
            17: 0.05,
            18: 0.05,
            19: 0.05,
            20: 0.05,
            21: 0.05,
            22: 0.05,
            23: 0.05,
        }
        self.spot_instance_removal_probability = {
            0: 0.05,
            1: 0.05,
            2: 0.05,
            3: 0.05,
            4: 0.05,
            5: 0.05,
            6: 0.05,
            7: 0.05,
            8: 0.05,
            9: 0.05,
            10: 0.05,
            11: 0.05,
            12: 0.05,
            13: 0.05,
            14: 0.05,
            15: 0.05,
            16: 0.05,
            17: 0.05,
            18: 0.05,
            19: 0.05,
            20: 0.05,
            21: 0.05,
            22: 0.05,
            23: 0.05,
        }
 
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

    def generate_spot_instance_add_event(self, delta):
        name = self.get_spot_instance_next_name()
        return {
            'kind': 'spot_instance_add',
            'delta': delta // self.millisecond,
            'name': name,
        }

    def generate_spot_instance_remove_event(self, delta, name):
        return {
            'kind': 'spot_instance_remove',
            'delta': delta // self.millisecond,
            'name': name,
        }

    def generate_spot_instance_events(self, start, duration):
        events = []
        millisecond = datetime.timedelta(milliseconds=1)
        current_delta = datetime.timedelta(seconds=0)
        current_instances = {}

        for i in range(self.spot_instance_desired_capacity):
            if self.spot_instance_initial_probability > self.r.random():
                event = self.generate_spot_instance_add_event(current_delta)
                events.append(event)
                current_instances[event['name']] = event['delta']

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

                    event = self.generate_spot_instance_remove_event(delta, name)
                    events.append(event)

                    del current_instances[name]
                # The next event is an addition, only do it if we're under desired capacity
                else:
                    delta = heapq.heappop(possible_added_deltas)

                    # Skip this addition
                    if len(current_instances) == self.spot_instance_desired_capacity:
                        continue

                    event = self.generate_spot_instance_add_event(delta)
                    events.append(event)

                    name = event['name']
                    current_instances[name] = delta

                    # Check if we also attempt to remove this new instance in this hour
                    delta = self.generate_spot_instance_removal_delta(current_time, current_delta)
                    if delta is None or delta < current_instances[name]:
                        continue
                    heapq.heappush(removed_instances, (delta, name))

            current_delta += datetime.timedelta(hours=1)

        return events

    def simulate(self):
        start = datetime.datetime.now(datetime.timezone.utc)
        start = start.replace(minute=0, second=0, microsecond=0)
        duration = datetime.timedelta(days=2)
        end = start + duration

        logger.info(f'Starting at {start}')

        logger.info(f'Generating spot instance events...')
        spot_instance_events = self.generate_spot_instance_events(start, duration)

        logger.info(f'Ending at {end}')
