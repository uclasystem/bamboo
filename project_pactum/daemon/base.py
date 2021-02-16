import os

class BaseDaemon:

	def __init__(self):
		self.run_dir = os.path.join(os.environ['XDG_RUNTIME_DIR'], 'project-pactum')
		self.sock_path = os.path.join(self.run_dir, 'daemon.sock')
