import sys

from daemon import DaemonContext
from project_pactum.daemon import ClientDaemon, ServerDaemon

HELP = "show the program's daemon command"

def add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='subcommand', dest='subcommand_name')

	start_parser = subparsers.add_parser('start', help='start help')
	start_parser.set_defaults(command=handle_start)
	start_parser.add_argument('--no-detach', action='store_true')

	stop_parser = subparsers.add_parser('stop', help='stop help')
	stop_parser.set_defaults(command=handle_stop)

def handle_start(options):
	daemon = ServerDaemon()
	context = DaemonContext(signal_map=daemon.signal_map)
	if options.no_detach:
		context.detach_process = False
		context.stderr = sys.stderr
		context.stdout = sys.stdout
	with context:
		daemon.start()

def handle_stop(options):
	daemon = ClientDaemon()
	print(daemon.get_reply('daemon stop'))
