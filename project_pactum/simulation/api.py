import argparse
import json
import logging
import random
import statistics

from project_pactum.simulation.simulator import Simulator

logger = logging.getLogger('project_pactum.simulation')

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--start-hour', type=int, choices=range(24))
    parser.add_argument('--generate-graphs', action='store_true')
    parser.add_argument('--generate-table', action='store_true')
    return parser.parse_args(args)

def main(args):
    from project_pactum.core.base import setup_logging
    setup_logging()

    options = parse(args)

    assert not (options.generate_graphs and options.generate_table)

    if not options.generate_table:
        simulator = Simulator(options)
        simulator.simulate()
    else:
        seed = options.seed
        if seed is not None:
            r = random.Random(seed)
            logger.info(f'Using seed: {seed}')
        else:
            r = random.Random()

        logging.getLogger('project_pactum.simulation.simulator').setLevel(logging.WARNING)

        all_preemptions = []
        all_fatal_failures = []
        all_instances = []
        all_performance = []
        all_cost = []
        all_value = []

        min_result = None
        max_result = None
        for _ in range(20):
            base_addition_probability = r.random()
            base_removal_probability = r.random()
            simulator = Simulator(options, base_addition_probability, base_removal_probability)
            result = simulator.simulate()
            if min_result is None:
                min_result = result
            elif result.num_preemptions < min_result.num_preemptions:
                min_result = result
            if max_result is None:
                max_result = result
            elif result.num_preemptions > max_result.num_preemptions:
                max_result = result
            all_preemptions.append(result.num_preemptions)
            all_fatal_failures.append(result.num_fatal_failures)
            all_instances.append(result.average_instances)
            all_performance.append(result.average_performance)
            all_cost.append(result.average_cost)
            all_value.append(result.average_value)

        print('       ', 'Preemptions', 'Fatal Failures', 'Instances', 'Performance', '     Cost', '    Value')
        print('Minimum',
            '{:11}'.format(min_result.num_preemptions),
            '{:14}'.format(min_result.num_fatal_failures),
            '{:9.2f}'.format(min_result.average_instances),
            '{:11.2f}'.format(min_result.average_performance),
            '{:9.2f}'.format(min_result.average_cost),
            '{:9.2f}'.format(min_result.average_value),
        )
        print('Average',
            '{:11}'.format(statistics.mean(all_preemptions)),
            '{:14}'.format(statistics.mean(all_fatal_failures)),
            '{:9.2f}'.format(statistics.mean(all_instances)),
            '{:11.2f}'.format(statistics.mean(all_performance)),
            '{:9.2f}'.format(statistics.mean(all_cost)),
            '{:9.2f}'.format(statistics.mean(all_value)),
        )
        print('Maximum',
            '{:11}'.format(max_result.num_preemptions),
            '{:14}'.format(max_result.num_fatal_failures),
            '{:9.2f}'.format(max_result.average_instances),
            '{:11.2f}'.format(max_result.average_performance),
            '{:9.2f}'.format(max_result.average_cost),
            '{:9.2f}'.format(max_result.average_value),
        )
