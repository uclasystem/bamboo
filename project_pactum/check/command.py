import platform

import project_pactum

HELP = "show program's check command"

def add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='command')

	from project_pactum.check.command import version_command
	version_parser = subparsers.add_parser('version', help=None)
	version_parser.set_defaults(command=version_command)

def version_command(options):
	print('Project Pactum:', project_pactum.__version__)
	print('Python:', platform.python_version())
