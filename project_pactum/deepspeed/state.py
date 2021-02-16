import subprocess

import project_pactum

from collections import OrderedDict
from deepspeed.launcher.runner import encode_world_info

class DeepspeedState:

	def __init__(self):
		self.processes = []

	def heartbeat(self):
		failed = False

                # Reverse the list so we can remove elements while iterating
		for process in reversed(self.processes):
			returncode = process.poll()
			if returncode is None:
				continue
			self.processes.remove(process)
			pid = process.pid
			print(f'PID {pid} finished with {returncode}')
			if returncode != 0:
				failed = True

		if failed:
			self.cleanup()

	def cleanup(self):
		for process in reversed(self.processes):
			self.processes.remove(process)
			process.terminate()

	def add(self):
		if self.processes:
			return 'Job already running'

		SSH_USERNAME = project_pactum.settings.SSH_USERNAME
		SSH_KEY = project_pactum.settings.SSH_KEY

		public_ips = ['3.239.49.200', '3.236.113.34']
		# public_ips = ['3.239.49.200']
		private_ip = '172.31.78.234'

		active_resources = OrderedDict()
		for public_ip in public_ips:
			active_resources[public_ip] = [0]

		world_info_base64 = encode_world_info(active_resources)

		for i, public_ip in enumerate(public_ips):
			example_path = '/home/project-pactum/src/external/deepspeed/DeepSpeedExamples/cifar'
			deepspeed_launch = [
				'export PYTHONPATH=/home/project-pactum/src/external/deepspeed',
				"&&",
				"cd {}".format(example_path),
				"&&",
				'python',
				"-u",
				"-m",
				"deepspeed.launcher.launch",
				f'--world_info={world_info_base64}',
				f"--node_rank={i}",
				f"--master_addr={private_ip}",
				"--master_port=29500",
			]
			example_deepspeed =  deepspeed_launch + [
				'cifar10_deepspeed.py',
				'--deepspeed_config',
				'ds_config.json',
			]
			cmd = [
				'ssh',
				'-i', SSH_KEY,
				f'{SSH_USERNAME}@{public_ip}',
				' '.join(example_deepspeed),
			]
			process = subprocess.Popen(cmd)
			self.processes.append(process)
		return 'Job added'

	def show(self):
		if not self.processes:
			return 'No running processes'
		ss = []
		for process in self.processes:
			ss.append('PID {}: {}'.format(process.pid, ' '.join(process.args)))
		return '\n'.join(ss)
