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

	def add_gpt2(self):
		if self.processes:
			return 'Job already running'

		SSH_USERNAME = project_pactum.settings.SSH_USERNAME
		SSH_KEY = project_pactum.settings.SSH_KEY

		public_ips = ['44.192.25.254', '3.236.23.138']
		private_ip = '172.31.70.66'

		active_resources = OrderedDict()
		for public_ip in public_ips:
			active_resources[public_ip] = [0]

		world_info_base64 = encode_world_info(active_resources)

		checkpoint_path = '/mnt/efs/checkpoints/gpt2_345m_ds'
		for i, public_ip in enumerate(public_ips):
			gpt_path = '/home/project-pactum/src/external/deepspeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism'
			deepspeed_launch = [
				'export PYTHONPATH=/home/project-pactum/src/external/deepspeed',
				"&&",
				"cd {}".format(gpt_path),
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
			gpt_deepspeed = deepspeed_launch + [
				'pretrain_gpt2.py',
				'--model-parallel-size', f'{project_pactum.settings.MP_SIZE}',
				'--pipe-parallel-size', f'{project_pactum.settings.PP_SIZE}',
				'--num-layers', f'{project_pactum.settings.NLAYERS}',
				'--hidden-size', f'{project_pactum.settings.NHIDDEN}',
				'--num-attention-heads', '16',
				'--seq-length', '1024',
				'--max-position-embeddings', '1024',
				'--batch-size', f'{project_pactum.settings.BATCHSIZE}',
				'--gas', '16',
				'--train-iters', f'{project_pactum.settings.TRAIN_ITERS}',
				'--lr-decay-iters', f'{project_pactum.settings.TRAIN_ITERS}',
				'--save', f'{project_pactum.settings.CHECKPOINT_PATH}',
				'--load', f'{project_pactum.settings.CHECKPOINT_PATH}',
				'--data-path', f'{project_pactum.settings.DATA_PATH}',
				'--vocab-file', f'{project_pactum.settings.VOCAB_PATH}',
				'--merge-file', f'{project_pactum.settings.MERGE_PATH}',
				'--data-impl', 'mmap',
				'--split', '949,50,1',
				'--distributed-backend', 'nccl',
				'--lr', '1.5e-4',
				'--lr-decay-style', 'cosine',
				'--min-lr', '1.0e-5',
				'--weight-decay', '1e-2',
				'--clip-grad', '1.0',
				'--warmup', '0.01',
				'--checkpoint-activations',
				'--log-interval', '1',
				'--save-interval', '10',
				'--eval-interval', '10',
				'--eval-iters', '5',
				'--fp16',
				'--tensorboard-dir', f'{project_pactum.settings.LOGDIR}',
				'--deepspeed',
				'--deepspeed_config', f'{project_pactum.settings.DS_CONFIG}',
				'--zero-stage', f'{project_pactum.settings.ZERO_STAGE}',
				'--zero-reduce-bucket-size', '50000000',
				'--zero-allgather-bucket-size', '5000000000',
				'--zero-contigious-gradients',
				'--zero-reduce-scatter',
				'--checkpoint-activations',
				'--checkpoint-num-layers', '1',
				'--partition-activations',
				'--synchronize-each-layer',
				'--contigious-checkpointing',
			]
			cmd = [
				'ssh',
				'-i', SSH_KEY,
				f'{SSH_USERNAME}@{public_ip}',
				' '.join(gpt_deepspeed),
			]
			process = subprocess.Popen(cmd)
			self.processes.append(process)

		return 'GPT2 job added'

	def show(self):
		if not self.processes:
			return 'No running processes'
		ss = []
		for process in self.processes:
			ss.append('PID {}: {}'.format(process.pid, ' '.join(process.args)))
		return '\n'.join(ss)
