import socket

from project_pactum.daemon import BaseDaemon

class ClientDaemon(BaseDaemon):

	def __init__(self):
		super().__init__()

	def get_reply(self, msg):
		sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		sock.connect(self.sock_path)
		sock.sendall(bytes(msg, 'utf-8'))
		reply = str(sock.recv(4096), 'utf-8')
		sock.close()
		return reply
