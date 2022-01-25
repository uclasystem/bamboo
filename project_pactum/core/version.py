import os
import subprocess

def get_version():
    from project_pactum import BASE_DIR
    version_script_path = os.path.join(BASE_DIR, 'version.sh')
    if os.path.exists(version_script_path):
        return subprocess.run([version_script_path],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True,
                              encoding='utf-8').stdout.strip()
    with open(os.path.join(BASE_DIR, 'VERSION'), 'r') as f:
        return f.read().strip()

def get_python_version(version):
    from re import match
    m = match('(\d+\.\d+\.\d+)-?(\d+)?', version)
    if m.group(2) is None:
        return m.group(1)
    return '{}.dev{}'.format(m.group(1), m.group(2))