import platform

import project_pactum

def test_command(options):
	import tensorflow as tf
	print('Project Pactum:', project_pactum.__version__)
	print('Python:', platform.python_version())
	print('Tensorflow:', tf.__version__)
