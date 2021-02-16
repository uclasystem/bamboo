from project_pactum.daemon import ClientDaemon

HELP = "show the program's deepspeed command"

def add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='subcommand', dest='subcommand_name')

	parser_show = subparsers.add_parser('show', help='print all deepspeed jobs')
	parser_show.set_defaults(command=handle_show)

	parser_add = subparsers.add_parser('add', help='create a deepspeed job')
	parser_add.set_defaults(command=handle_add)

def handle_add(options):
	client = ClientDaemon()
	print(client.get_reply('deepspeed add'))

def handle_show(options):
	client = ClientDaemon()
	print(client.get_reply('deepspeed show'))
