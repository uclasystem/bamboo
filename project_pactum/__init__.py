import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = subprocess.run([os.path.join(BASE_DIR, 'version.sh')],
                         capture_output=True,
                         check=True,
                         encoding='utf-8').stdout.strip()

def main(args):
	from project_pactum.core.base import parse, setup

	options = parse(args)
	setup()
