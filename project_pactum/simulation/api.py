import argparse
import json

from project_pactum.simulation.simulator import Simulator

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--start-hour', type=int, choices=range(24))
    parser.add_argument('--generate-graphs', action='store_true')
    return parser.parse_args(args)

def main(args):
    from project_pactum.core.base import setup_logging
    setup_logging()

    options = parse(args)

    simulator = Simulator(options)
    simulator.simulate()
