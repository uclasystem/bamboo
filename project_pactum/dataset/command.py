import logging

import project_pactum

from project_pactum.dataset.base import slug_to_var_name

HELP = None

logger = logging.getLogger(__name__)

def add_arguments(parser):
	from project_pactum.dataset.command import add_command, list_command, remove_command
	subparsers = parser.add_subparsers(dest='subcommand', required=True)

	# subparsers = parser.add_subparsers(title='subcommands', dest='subcommand', required=True)

	add_parser = subparsers.add_parser('add', help=None)
	add_parser.set_defaults(command=add_command)
	add_parser.add_argument('datasets', nargs='+')

	list_parser = subparsers.add_parser('list', help=None)
	list_parser.set_defaults(command=list_command)

	remove_parser = subparsers.add_parser('remove', help=None)
	remove_parser.set_defaults(command=remove_command)
	remove_parser.add_argument('datasets', nargs='+')

def add_command(options):
	for slug in options.datasets:
		var_name = slug_to_var_name(slug)
		dataset = project_pactum.datasets[var_name]
		if not dataset.exists():
			dataset.add()
		else:
			logger.warn('{} already exists'.format(slug))

def list_command(options):
	for _, dataset in sorted(project_pactum.datasets.items()):
		if dataset.exists():
			marker = '[*]'
		else:
			marker = '[ ]'
		print(marker, dataset.SLUG)

def remove_command(options):
	for slug in options.datasets:
		var_name = slug.replace('-', '_')
		dataset = project_pactum.datasets[var_name]
		if dataset.exists():
			dataset.remove()
		else:
			logger.warn("{} doesn't exist".format(slug))
