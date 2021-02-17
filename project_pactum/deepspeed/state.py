import subprocess

import project_pactum
import project_pactum.aws.instance

from collections import OrderedDict
from deepspeed.launcher.runner import encode_world_info

class DeepspeedState:

	def __init__(self):
		self.processes = []
		self.instances = {}

	def instance_heartbeat(self, name):
		current_instances = {}
		for instance in project_pactum.aws.instance.get_instances():
			if 'PublicIpAddress' not in instance:
				continue
			if instance['State']['Name'] != 'running':
				continue
			if 'Tags' not in instance:
				continue
			found_tag = False
			for tag in instance['Tags']:
				if tag['Key'] == 'Name' and tag['Value'] == name:
					found_tag = True
					break
			if not found_tag:
				continue
			current_instances[instance['InstanceId']] = instance

		added_instances = current_instances.keys() - self.instances.keys()
		if added_instances:
			self.added_instances_listener(added_instances)
		removed_instances = self.instances.keys() - current_instances.keys()
		if removed_instances:
			self.removed_instances_listener(removed_instances)
		self.instances = current_instances


	def added_instances_listener(self, instance_ids):
		pass

	def removed_instances_listener(self, instance_ids):
		pass

	def heartbeat(self):
		self.instance_heartbeat(project_pactum.settings.AWS_NAME)

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

		public_ips = []
		active_resources = OrderedDict()
		for i, instance in enumerate(self.instances.values()):
			private_ip = instance['PrivateIpAddress']
			if i == 0:
				master_addr = private_ip
			# Assume one GPU per node right now, use InstanceType later
			active_resources[private_ip] = [0]

			public_ip = instance['PublicIpAddress']
			public_ips.append(public_ip)
		if not public_ips:
			return 'No running instances'

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
				f"--master_addr={master_addr}",
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
