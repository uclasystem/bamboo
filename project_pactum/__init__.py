import functools
import logging
import os
import shlex
import subprocess

from project_pactum.version import get_version, get_python_version

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERSION = get_version()
__version__ = get_python_version(VERSION)

@functools.wraps(subprocess.run)
def run(args, **kwargs):
	logger = logging.getLogger('project_pactum.run')
	logger.debug(shlex.join(args))
	p = subprocess.run(args, **kwargs)
	return p

def main(args):
	from project_pactum.core.base import parse, setup_logging

	setup_logging()

	options = parse(args)

	if 'command' in options:
		options.command(options)
