import argparse
import functools
import importlib
import logging
import pkgutil
import subprocess

import project_pactum

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

def parse(args):
	parser = argparse.ArgumentParser(prog='project_pactum',
	                                 description='Project Pactum')

	from project_pactum import VERSION
	parser.add_argument('--version', action='version',
	                    version='Project Pactum {}'.format(VERSION))

	subparsers = parser.add_subparsers(metavar='command', dest='command_name')

	for module_info in pkgutil.iter_modules(project_pactum.__path__):
		if not module_info.ispkg:
			continue
		full_module_name = 'project_pactum.{}.command'.format(module_info.name)
		try:
			module = importlib.import_module(full_module_name)
		except ModuleNotFoundError as e:
			continue

		subparser = subparsers.add_parser(module_info.name, help=module.HELP)
		module.add_arguments(subparser)

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

def setup_settings():
	from project_pactum.core.settings import Settings
	settings = Settings()
	setattr(project_pactum, 'settings', settings)

@functools.wraps(subprocess.run)
def run(args, **kwargs):
	logger = logging.getLogger('project_pactum.run')
	logger.debug(' '.join(args))
	p = subprocess.run(args, **kwargs)
	return p

def setup(options):
	setup_logging()
	setup_settings()

	if options.command_name == 'dataset':
		from project_pactum.dataset.base import setup_datasets
		setup_datasets()
