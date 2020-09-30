import os

import project_pactum.dataset.base as base

from project_pactum.core.base import run

class Cifar10(base.Dataset):
	SLUG = 'cifar-10'
	NAME = 'CIFAR-10'
	DATASET_DIR = os.path.join(base.Dataset.BASE_DIR, SLUG)

	def exists(self):
		return os.path.exists(self.DATASET_DIR)

	def add(self):
		os.makedirs(self.DATASET_DIR, exist_ok=True)
		tar_path = os.path.join(self.BASE_DIR, '{}.tar.gz'.format(self.SLUG))
		run(['wget', 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
		     '-O', tar_path],
		    check=True, cwd=self.DATASET_DIR)
		run(['tar', 'xzf', tar_path, '-C', self.DATASET_DIR, '--strip-components=1'],
		    check=True, cwd=self.DATASET_DIR)
		run(['rm', tar_path], check=True)

	def remove(self):
		run(['rm', '-rf', self.DATASET_DIR], check=True)
