import argparse
import functools
import logging
import subprocess

from project_pactum import VERSION

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
	parser.add_argument('--version', action='version',
	                    version='Project Pactum {}'.format(VERSION))
	return parser.parse_args(args)

def setup_logging():
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(ProjectPactumFormatter())
	logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler])

def setup():
	from project_pactum.dataset.base import setup_datasets

	setup_logging()
	setup_datasets()

@functools.wraps(subprocess.run)
def run(args, **kwargs):
	logger = logging.getLogger('project_pactum.run')
	logger.debug(' '.join(args))
	p = subprocess.run(args, **kwargs)
	return p
