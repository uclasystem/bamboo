import argparse
import json

from project_pactum.simulation.simulator import Simulator

def parse(args):
    parser = argparse.ArgumentParser()
    return parser.parse_args(args)

def main(args):
    from project_pactum.core.base import setup_logging
    setup_logging()

    options = parse(args)

    simulator = Simulator()
    simulator.simulate()
