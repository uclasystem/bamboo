import platform

import project_pactum

def version_command(options):
	import tensorflow as tf
	print('Python:', platform.python_version())
	print('Tensorflow:', tf.__version__)
	print('Project Pactum:', project_pactum.__version__)
