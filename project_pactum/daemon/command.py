import daemon
import functools
import os
import pid
import select
import signal
import socket
import sys
import threading

from time import sleep

import project_pactum

SOCK_PATH = os.path.join(pid.DEFAULT_PID_DIR, 'project-pactum.sock')

def signal_handler(coordinator, signum, frame):
	coordinator.shutdown()
	sys.exit(0)

def socket_loop(coordinator):
	try:
		os.unlink(SOCK_PATH)
	except:
		if os.path.exists(SOCK_PATH):
			raise
	sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	sock.bind(SOCK_PATH)
	sock.listen(1)
	while coordinator.is_running():
		client_sock, _ = sock.accept()
		msg = str(client_sock.recv(4096), 'utf-8')
		reply = coordinator.get_reply(msg)
		client_sock.sendall(bytes(reply, 'utf-8'))
		client_sock.close()
	sock.close()

def main_loop(coordinator):
	x = threading.Thread(target=socket_loop, args=(coordinator,), daemon=True)
	x.start()

	while coordinator.is_running():
		coordinator.check_cloudwatch()
		coordiantor.check_terminated()
		coordinator.ensure_count()
		sleep(5)

def main_command(options):
	from project_pactum.daemon.coordinator import Coordinator
	coordinator = Coordinator(options.count, options.instance_type,
	                          options.zone)

	pidfile = pid.PidFile('project-pactum')
	handler = functools.partial(signal_handler, coordinator)
	signal_map = {
		signal.SIGINT: handler,
		signal.SIGTERM: handler,
	}
	context = daemon.DaemonContext(files_preserve=[coordinator.csv_file],
	                               pidfile=pidfile,
	                               signal_map=signal_map)
	if options.debug:
		context.detach_process = False
		context.stderr = sys.stderr
		context.stdout = sys.stdout
	with context:
		main_loop(coordinator)

def list_command(options):
	sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	sock.connect(SOCK_PATH)
	sock.sendall(bytes('list', 'utf-8'))
	msg = str(sock.recv(4096), 'utf-8')
	print(msg)
	sock.close()

def test_command(options):
	sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	sock.connect(SOCK_PATH)
	sock.sendall(bytes('test', 'utf-8'))
	msg = str(sock.recv(4096), 'utf-8')
	print(msg)
	sock.close()
