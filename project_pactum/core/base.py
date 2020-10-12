import argparse
import functools
import logging
import subprocess

class ProjectPactumFormatter(logging.Formatter):

	def format(self, record):
		COLORS = {
			logging.DEBUG: 35,
			logging.INFO: 36,
			logging.WARNING: 33,
			logging.ERROR: 31,
		}
		fmt = '[\x1B[1;{color}m%(levelname)s\x1B[m \x1B[{color}m%(name)s\x1B[m] %(message)s'
		formatter = logging.Formatter(fmt.format(color=COLORS[record.levelno]))
		return formatter.format(record)

def core_add_arguments(parser):
	from project_pactum import VERSION
	parser.add_argument('--version', action='version',
	                    version='Project Pactum {}'.format(VERSION))

def aws_add_arguments(parser):
	from project_pactum.aws.command import test_command
	subparsers = parser.add_subparsers(metavar='command')

	test_parser = subparsers.add_parser('test', help=None)
	test_parser.set_defaults(command=test_command)

def check_add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='command')

	from project_pactum.check.command import version_command
	version_parser = subparsers.add_parser('version', help=None)
	version_parser.set_defaults(command=version_command)

def dataset_add_arguments(parser):
	from project_pactum.dataset.command import add_command, list_command, remove_command
	subparsers = parser.add_subparsers(metavar='command')

	add_parser = subparsers.add_parser('add', help=None)
	add_parser.set_defaults(command=add_command)
	add_parser.add_argument('datasets', nargs='+')

	list_parser = subparsers.add_parser('list', help=None)
	list_parser.set_defaults(command=list_command)

	remove_parser = subparsers.add_parser('remove', help=None)
	remove_parser.set_defaults(command=remove_command)
	remove_parser.add_argument('datasets', nargs='+')

def experiment_add_arguments(parser):
	from project_pactum.experiment.command import tutorial_mnist_command
	subparsers = parser.add_subparsers(metavar='command')

	tutorial_mnist_parser = subparsers.add_parser('tutorial-mnist', help=None)
	tutorial_mnist_parser.set_defaults(command=tutorial_mnist_command)

	tutorial_mnist_parser.add_argument('--host', action='store_true')

def parse(args):
	parser = argparse.ArgumentParser(prog='project_pactum',
	                                 description='Project Pactum')
	core_add_arguments(parser)

	subparsers = parser.add_subparsers(metavar='command')

	aws_parser = subparsers.add_parser('aws', help=None)
	aws_add_arguments(aws_parser)

	check_parser = subparsers.add_parser('check', help=None)
	check_add_arguments(check_parser)

	dataset_parser = subparsers.add_parser('dataset', help=None)
	dataset_add_arguments(dataset_parser)

	experiment_parser = subparsers.add_parser('experiment', help=None)
	experiment_add_arguments(experiment_parser)

	return parser.parse_args(args)

def setup_logging():
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(ProjectPactumFormatter())
	logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler])

	logging.getLogger('botocore.parsers').addHandler(logging.NullHandler())
	logging.getLogger('botocore.parsers').propagate = False

	logging.getLogger('absl').setLevel(logging.ERROR)
	logging.getLogger('tensorflow').setLevel(logging.ERROR)

	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_tensorflow():
	import tensorflow as tf
	gpus = tf.config.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

def setup():
	from project_pactum.dataset.base import setup_datasets

	setup_logging()
	setup_datasets()
	setup_tensorflow()

@functools.wraps(subprocess.run)
def run(args, **kwargs):
	logger = logging.getLogger('project_pactum.run')
	logger.debug(' '.join(args))
	p = subprocess.run(args, **kwargs)
	return p
