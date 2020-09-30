import importlib
import inspect
import os
import pkgutil

import project_pactum

def slug_to_var_name(slug):
	return slug.replace('-', '_')

class Dataset:
	BASE_DIR = os.path.join(project_pactum.BASE_DIR, 'dataset')

	def get_var_name(self):
		return slug_to_var_name(self.SLUG)

def setup_datasets():
	datasets = {}
	setattr(project_pactum, 'datasets', datasets)

	dirname = os.path.dirname(__file__)
	for _, name, _ in pkgutil.iter_modules([dirname]):
		if name in ['base', 'command']:
			continue

		module = importlib.import_module('project_pactum.dataset.{}'.format(name))

		for _, cls in inspect.getmembers(module, inspect.isclass):
			dataset = cls()
			datasets[dataset.get_var_name()] = dataset
