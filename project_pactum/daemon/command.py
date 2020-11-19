import daemon
import multiprocessing
import os
import pid
import select
import socket
import sys

from time import sleep

import project_pactum
import project_pactum.experiment.imagenet_pretrain

SOCK_PATH = os.path.join(pid.DEFAULT_PID_DIR, 'project-pactum.sock')

def sock_recv(sock):
	chunk = sock.recv(4096)
	print("Got:", chunk, len(chunk))

def main_loop(options):
	from project_pactum.daemon.coordinator import Coordinator
	coordinator = Coordinator()

	try:
		os.unlink(SOCK_PATH)
	except:
		if os.path.exists(SOCK_PATH):
			raise
	sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	sock.setblocking(False)
	sock.bind(SOCK_PATH)
	sock.listen(1)

	## Temporarily hardcoding imagenet experiment
	coordinator.active_servers = imagenet_pretrain.run(options)
	while True:
		ready_to_read, _, _ = select.select([sock], [], [], 0)
		for ready_sock in ready_to_read:
			client_sock, _ = ready_sock.accept()
			sock_recv(client_sock)
		coordinator.check_cloudwatch()
		status = imagenet_pretrain.status(options, coordinator.active_servers)
		if "imagenet-pretrain finished" in status:
			break
		sleep(5)
		# sock.close()

	imagenet_pretrain.shutdown(coordinator.active_servers)


def main_command(options):
	pidfile = pid.PidFile('project-pactum')
	context = daemon.DaemonContext(pidfile=pidfile)
	if options.debug:
		context.detach_process = False
		context.stderr = sys.stderr
		context.stdout = sys.stdout
	with context:
		main_loop(options)

def test_command(options):
	sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	sock.connect(SOCK_PATH)
	sock.sendall(b'test\0')
	sock.close()
