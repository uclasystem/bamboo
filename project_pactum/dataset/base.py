import importlib
import inspect
import os
import pkgutil

import project_pactum

class Dataset:
	BASE_DIR = os.path.join(project_pactum.BASE_DIR, 'dataset')

	def get_var_name(self):
		return self.SLUG.replace('-', '_')

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
