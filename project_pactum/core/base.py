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
	parser.add_argument('--daemonize', action='store_true')

def aws_add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='command')

	from project_pactum.aws.command import test_command
	test_parser = subparsers.add_parser('test', help=None)
	test_parser.set_defaults(command=test_command)

	from project_pactum.aws.command import add_command
	add_parser = subparsers.add_parser('add', help=None)
	add_parser.set_defaults(command=add_command)

	from project_pactum.aws.command import cloudwatch_command
	cloudwatch_parser = subparsers.add_parser('cloudwatch', help=None)
	cloudwatch_parser.set_defaults(command=cloudwatch_command)

	from project_pactum.aws.command import list_command
	list_parser = subparsers.add_parser('list', help=None)
	list_parser.set_defaults(command=list_command)

	from project_pactum.aws.command import terminate_command
	terminate_parser = subparsers.add_parser('terminate', help=None)
	terminate_parser.set_defaults(command=terminate_command)
	terminate_parser.add_argument('instance_ids', metavar='instance-id', nargs='+')

def check_add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='command')

	from project_pactum.check.command import version_command
	version_parser = subparsers.add_parser('version', help=None)
	version_parser.set_defaults(command=version_command)

def control_add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='command')

	from project_pactum.daemon.command import list_command
	list_parser = subparsers.add_parser('list', help=None)
	list_parser.set_defaults(command=list_command)

	from project_pactum.daemon.command import test_command
	test_parser = subparsers.add_parser('test', help=None)
	test_parser.set_defaults(command=test_command)

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

def daemon_add_arguments(parser):
	from project_pactum.daemon.command import main_command

	parser.set_defaults(command=main_command)
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--count', type=int, default=8)
	parser.add_argument('--instance-type', type=str, default='p2.xlarge')
	parser.add_argument('--zone', type=str, default='us-east-1d')

def experiment_add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='command')

	from project_pactum.experiment.command import aws_availability_command
	aws_availability_parser = subparsers.add_parser('aws-availability', help=None)
	aws_availability_parser.set_defaults(command=aws_availability_command)
	aws_availability_parser.add_argument('--skip-monitor', action='store_true')
	aws_availability_parser.add_argument('--analyze-daily', action='store_true')

	from project_pactum.experiment.command import test_command
	test_parser = subparsers.add_parser('test', help=None)
	test_parser.set_defaults(command=test_command)

	from project_pactum.experiment.command import tutorial_mnist_command
	tutorial_mnist_parser = subparsers.add_parser('tutorial-mnist', help=None)
	tutorial_mnist_parser.set_defaults(command=tutorial_mnist_command)
	tutorial_mnist_parser.add_argument('--worker-index', type=int, default=0)

	from project_pactum.experiment.command import imagenet_pretrain_command
	imagenet_parser = subparsers.add_parser('imagenet-pretrain', help=None)
	imagenet_parser.set_defaults(command=imagenet_pretrain_command)
	imagenet_parser.add_argument('--cluster-size', type=int, default=1)
	imagenet_parser.add_argument('--instance-type', type=str, default='p2.xlarge')
	imagenet_parser.add_argument('--ngpus', type=int, default=1)
	imagenet_parser.add_argument('--az', type=str, default=None)
	imagenet_parser.add_argument('--epochs', type=int, default=1)

def parse(args):
	parser = argparse.ArgumentParser(prog='project_pactum',
									 description='Project Pactum')
	core_add_arguments(parser)

	subparsers = parser.add_subparsers(metavar='command', dest='command_name')

	aws_parser = subparsers.add_parser('aws', help=None)
	aws_add_arguments(aws_parser)

	check_parser = subparsers.add_parser('check', help=None)
	check_add_arguments(check_parser)

	control_parser = subparsers.add_parser('control', help=None)
	control_add_arguments(control_parser)

	dataset_parser = subparsers.add_parser('dataset', help=None)
	dataset_add_arguments(dataset_parser)

	daemon_parser = subparsers.add_parser('daemon', help=None)
	daemon_add_arguments(daemon_parser)

	experiment_parser = subparsers.add_parser('experiment', help=None)
	experiment_add_arguments(experiment_parser)

	return parser.parse_args(args)

def setup_logging():
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(ProjectPactumFormatter())
	logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler])

	logging.getLogger('botocore.auth').setLevel(logging.WARNING)
	logging.getLogger('botocore.client').setLevel(logging.WARNING)
	logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
	logging.getLogger('botocore.endpoint').setLevel(logging.WARNING)
	logging.getLogger('botocore.handlers').setLevel(logging.WARNING)
	logging.getLogger('botocore.hooks').setLevel(logging.WARNING)
	logging.getLogger('botocore.loaders').setLevel(logging.WARNING)
	logging.getLogger('botocore.parsers').setLevel(logging.WARNING)
	logging.getLogger('botocore.retryhandler').setLevel(logging.WARNING)
	logging.getLogger('botocore.utils').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.action').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.collection').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.factory').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.model').setLevel(logging.WARNING)

	logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

	logging.getLogger('PidFile').setLevel(logging.WARNING)

	logging.getLogger('absl').setLevel(logging.WARNING)
	logging.getLogger('tensorflow').setLevel(logging.WARNING)

	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_tensorflow():
	import tensorflow as tf
	gpus = tf.config.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

@functools.wraps(subprocess.run)
def run(args, **kwargs):
	logger = logging.getLogger('project_pactum.run')
	logger.debug(' '.join(args))
	p = subprocess.run(args, **kwargs)
	return p

def setup(options):
	setup_logging()

	if options.command_name == 'dataset':
		from project_pactum.dataset.base import setup_datasets
		setup_datasets()
