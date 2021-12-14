import argparse
import collections
import json
import logging
import multiprocessing
import random
import statistics

from project_pactum.simulation.simulator import Simulator

logger = logging.getLogger('project_pactum.simulation')

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--start-hour', type=int, choices=range(24))
    parser.add_argument('--generate-addition-probabilities', action='store_true')
    parser.add_argument('--removal-probability', type=float, default=None)
    parser.add_argument('--generate-graphs', action='store_true')
    parser.add_argument('--generate-table', action='store_true')
    parser.add_argument('--spot-instance-trace', type=argparse.FileType('r'), default=None)
    return parser.parse_args(args)

def graph(xlabel, xs, xmax, ylabel, ys, ymax, average,
          on_demand=None, out=None, show=False):
        import matplotlib.pyplot as plt

        # sizes: xx-small, x-small, small, medium, large, x-large, xx-large
        params = {
            'font.family': 'Inter',
            'legend.fontsize': 'x-small',
            'axes.labelsize': 'x-small',
            'axes.titlesize': 'x-small',
            'xtick.labelsize': 'x-small',
            'ytick.labelsize': 'x-small',
            'figure.figsize': (3.0, 1.7),
        }
        plt.rcParams.update(params)

        plt.clf()

        plt.plot(xs, ys, linewidth=0.5)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        for scale in [1, 2, 6, 12, 24, 48, 72]:
            xticks = list(range(0, xmax + 1, scale))
            if len(xticks) < 7:
                break
        if xmax not in xticks:
            xticks.append(xmax)
        plt.xticks(xticks)

        plt.xlim(0, xmax)
        plt.ylim(0, ymax)

        plt.hlines(average, 0, xmax, color='tab:blue', linestyles='dotted')
        if on_demand is not None:
            plt.hlines(on_demand, 0, xmax, color='tab:red', linestyles='dashed')

        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if out is not None:
           plt.savefig(
               out,
               bbox_inches='tight',
               pad_inches=0
           )

        if show:
            plt.show()

def simulate(args):
    removal_probability, seed = args
    simulator = Simulator(
        seed=seed,
        start_hour=0,
        generate_addition_probabilities=True,
        removal_probability=removal_probability
    )
    result = simulator.simulate()
    return result

def generate_table():
    logging.getLogger('project_pactum.simulation.simulator').setLevel(logging.WARNING)

    count = 0

    removal_probabilities = [0.01, 0.05, 0.10, 0.25, 0.50]
    all_preemptions = {}
    all_interruptions = {}
    all_lifetimes = {}
    all_fatal_failures = {}
    all_instances = {}
    all_performance = {}
    all_cost = {}
    all_value = {}
    for removal_probability in removal_probabilities:
        all_preemptions[removal_probability] = []
        all_interruptions[removal_probability] = []
        all_lifetimes[removal_probability] = []
        all_fatal_failures[removal_probability] = []
        all_instances[removal_probability] = []
        all_performance[removal_probability] = []
        all_cost[removal_probability] = []
        all_value[removal_probability] = []

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        simulations = []
        for removal_probability in removal_probabilities:
            for seed in range(1, 10_001):
                simulations.append((removal_probability, seed))

        for result in pool.imap_unordered(simulate, simulations):
            removal_probability = result.removal_probability
            all_preemptions[removal_probability].append(result.num_preemptions)
            all_interruptions[removal_probability].append(result.preemption_mean)
            all_lifetimes[removal_probability].append(result.lifetime_mean)
            all_fatal_failures[removal_probability].append(result.num_fatal_failures)
            all_instances[removal_probability].append(result.average_instances)
            all_performance[removal_probability].append(result.average_performance)
            all_cost[removal_probability].append(result.average_cost)
            all_value[removal_probability].append(result.average_value)

            count += 1
            if count % 100 == 0:
                logger.info(f'{count} simulations complete')

    print('Probability', 'Preemptions', 'Interruptions', 'Lifetimes', 'Fatal Failures', 'Instances', 'Performance', '     Cost', '    Value',
          sep=' & ', end=' \\\\\n')
    for removal_probability in removal_probabilities:
        print(f'{removal_probability:11.2f}',
            '{:11.2f}'.format(statistics.mean(all_preemptions[removal_probability])),
            '{:13.2f}'.format(statistics.mean(all_interruptions[removal_probability])),
            '{:9.2f}'.format(statistics.mean(all_lifetimes[removal_probability])),
            '{:14.2f}'.format(statistics.mean(all_fatal_failures[removal_probability])),
            '{:9.2f}'.format(statistics.mean(all_instances[removal_probability])),
            '{:11.2f}'.format(statistics.mean(all_performance[removal_probability])),
            '{:9.2f}'.format(statistics.mean(all_cost[removal_probability])),
            '{:9.2f}'.format(statistics.mean(all_value[removal_probability])),
            sep=' & ', end=' \\\\\n'
        )

def main(args):
    from project_pactum.core.base import setup_logging
    setup_logging()

    options = parse(args)

    assert not (options.generate_graphs and options.generate_table)

    if not options.generate_table:
        simulator = Simulator(
            seed=options.seed,
            start_hour=options.start_hour,
            generate_addition_probabilities=options.generate_addition_probabilities,
            removal_probability=options.removal_probability,
            generate_graphs=options.generate_graphs,
            spot_instance_trace=options.spot_instance_trace,
        )
        simulator.simulate(duration=43_200_000)
    else:
        generate_table()
