import os
import subprocess

def get_version():
	from project_pactum import BASE_DIR
	return subprocess.run([os.path.join(BASE_DIR, 'version.sh')],
	                      capture_output=True,
	                      check=True,
	                      encoding='utf-8').stdout.strip()


def get_python_version(version):
	from re import match
	m = match('(\d+\.\d+\.\d+)-?(\d+)?', version)
	if m.group(2) is None:
		return m.group(1)
	return '{}.dev{}'.format(m.group(1), m.group(2))
