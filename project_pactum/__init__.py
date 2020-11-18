import os
import subprocess

from daemon import DaemonContext

from project_pactum.core.version import get_version, get_python_version

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERSION = get_version()
__version__ = get_python_version(VERSION)

def main(args):
	from project_pactum.core.base import parse, setup

	options = parse(args)

	setup(options)

	if 'command' in options:
		if options.daemonize:
			import project_pactum
			with open(os.path.join(project_pactum.BASE_DIR, 'daemonize.txt'), 'w') as f:
				with DaemonContext(stdout=f, stderr=f):
					options.command(options)
		else:
			options.command(options)
