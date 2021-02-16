import os

from project_pactum.core.version import get_version, get_python_version

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERSION = get_version()
__version__ = get_python_version(VERSION)

def main(args):
	from project_pactum.core.base import parse, setup

	options = parse(args)

	setup(options)

	if 'command' in options:
		options.command(options)
