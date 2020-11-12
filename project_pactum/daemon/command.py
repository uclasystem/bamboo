import daemon
import os
import pid
import socket
import sys

from time import sleep

import project_pactum

def main_command(options):
	from project_pactum.daemon.coordinator import Coordinator
	coordinator = Coordinator()

	log_path = os.path.join(project_pactum.BASE_DIR, 'log.txt')
	sock_path = os.path.join(pid.DEFAULT_PID_DIR, 'project-pactum.sock')
	pidfile = pid.PidFile('project-pactum')
	context = daemon.DaemonContext(pidfile=pidfile)
	sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
	try:
		os.unlink(sock_path)
	except:
		pass
	sock.bind(sock_path)
	if options.debug:
		context.detach_process = False
		context.stderr = sys.stderr
		context.stdout = sys.stdout
	with context:
		while True:
			coordinator.check_cloudwatch()
			sleep(5)
		sock.close()
