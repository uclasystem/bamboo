import errno
import os
import signal
import socket
import sys
import threading
import time

import project_pactum

from project_pactum.daemon import BaseDaemon

from project_pactum.deepspeed.state import DeepspeedState

class ServerDaemon(BaseDaemon):

	def __init__(self):
		super().__init__()

		self.running = True
		self.signal_map = {
			signal.SIGINT: self.signal,
			signal.SIGTERM: self.signal,
		}

		self.states = {}
		self.states['deepspeed'] = DeepspeedState()

	# TODO: for now just shut down for any signal
	def signal(self, signum, frame):
		self.running = False

	def excepthook(self, args):
		self.running = False
		sys.excepthook(args.exc_type, args.exc_value, args.exc_traceback)
		print(f'Thread: {args.thread}', file=sys.stderr)

	def start(self):
		self.threads = [
			threading.Thread(name='unix_socket_listener', target=self.unix_socket_listener),
		]
		threading.excepthook = self.excepthook

		os.mkdir(self.run_dir)
		[t.start() for t in self.threads]
		while self.running:
			self.states['deepspeed'].heartbeat()
			time.sleep(1)
		self.states['deepspeed'].cleanup()
		[t.join() for t in self.threads]
		os.rmdir(self.run_dir)

	def unix_socket_listener(self):
		if os.path.exists(self.sock_path):
			raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), self.sock_path)
		sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		sock.bind(self.sock_path)
		sock.listen(1)
		sock.settimeout(1)
		while self.running:
			try:
				client_sock, _ = sock.accept()
			except socket.timeout:
				continue
			msg = str(client_sock.recv(4096), 'utf-8')
			reply = self.get_reply(msg)
			client_sock.sendall(bytes(reply, 'utf-8'))
			client_sock.close()
		sock.close()
		os.unlink(self.sock_path)

	def get_reply(self, msg):
		if msg == 'daemon stop':
			self.running = False
			return 'daemon stopping'
		elif msg == 'deepspeed add':
			return self.states['deepspeed'].add()
		elif msg == 'deepspeed add-local-instance':
			return self.states['deepspeed'].add_local_instance()
		elif msg == 'deepspeed add-gpt2':
			return self.states['deepspeed'].add_gpt2()
		elif msg == 'deepspeed show':
			return self.states['deepspeed'].show()
		else:
			return 'Not implemented'
