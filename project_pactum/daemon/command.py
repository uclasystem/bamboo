import daemon
import multiprocessing
import os
import pid
import select
import socket
import sys

from time import sleep

import project_pactum

SOCK_PATH = os.path.join(pid.DEFAULT_PID_DIR, 'project-pactum.sock')

def sock_recv(sock):
	chunk = sock.recv(4096)
	print("Got:", chunk, len(chunk))

def main_loop():
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
	while True:
		ready_to_read, _, _ = select.select([sock], [], [], 0)
		for ready_sock in ready_to_read:
			client_sock, _ = ready_sock.accept()
			sock_recv(client_sock)
		coordinator.check_cloudwatch()
		sleep(5)
		# sock.close()


def main_command(options):
	pidfile = pid.PidFile('project-pactum')
	context = daemon.DaemonContext(pidfile=pidfile)
	if options.debug:
		context.detach_process = False
		context.stderr = sys.stderr
		context.stdout = sys.stdout
	with context:
		main_loop()

def test_command(options):
	sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	sock.connect(SOCK_PATH)
	sock.sendall(b'test\0')
	sock.close()
