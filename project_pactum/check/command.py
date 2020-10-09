import platform
import tensorflow

import project_pactum

def test_command(options):
	print('Project Pactum:', project_pactum.__version__)
	print('Python:', platform.python_version())
	print('Tensorflow:', tensorflow.__version__)
