import logging

import project_pactum

logger = logging.getLogger(__name__)

def add_command(options):
	for slug in options.datasets:
		var_name = slug.replace('-', '_')
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
