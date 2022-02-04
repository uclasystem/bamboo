import argparse
import logging
import time

import project_pactum

from colorama import Fore, Style

logger = logging.getLogger(__name__)

class ProjectPactumFormatter(logging.Formatter):

	def __init__(self):
		self.created = time.time()

	def format(self, record):
		reltime = record.created - self.created
		COLORS = {
			logging.DEBUG: 35,
			logging.INFO: 36,
			logging.WARNING: 33,
			logging.ERROR: 31,
		}
		fmt = '\x1B[1;{color}m[{reltime:.3f} p%(process)d/t%(thread)d %(levelname)s %(name)s]\x1B[m \x1B[{color}m%(message)s\x1B[m'
		formatter = logging.Formatter(fmt.format(color=COLORS[record.levelno], reltime=reltime))
		return formatter.format(record)

def parse(args):
	parser = argparse.ArgumentParser(prog='project_pactum',
	                                 description='Project Pactum')

	parser.add_argument(
		'--version', action='version',
		version=f'{Fore.BLUE}{Style.BRIGHT}Bamboo{Style.RESET_ALL}'
		        f' {Style.BRIGHT}{project_pactum.__version__}{Style.RESET_ALL}')

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
	logging.getLogger('botocore.httpsession').setLevel(logging.WARNING)
	logging.getLogger('botocore.loaders').setLevel(logging.WARNING)
	logging.getLogger('botocore.parsers').setLevel(logging.WARNING)
	logging.getLogger('botocore.retryhandler').setLevel(logging.WARNING)
	logging.getLogger('botocore.utils').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.action').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.collection').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.factory').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.model').setLevel(logging.WARNING)

	logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

	logging.getLogger('matplotlib').setLevel(logging.WARNING)
