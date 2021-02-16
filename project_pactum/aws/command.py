NAME = 'aws'
HELP = None

def add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='command')

	from project_pactum.aws.command import test_command
	test_parser = subparsers.add_parser('test', help=None)
	test_parser.set_defaults(command=test_command)

	from project_pactum.aws.command import add_command
	add_parser = subparsers.add_parser('add', help=None)
	add_parser.set_defaults(command=add_command)

	from project_pactum.aws.command import cloudwatch_command
	cloudwatch_parser = subparsers.add_parser('cloudwatch', help=None)
	cloudwatch_parser.set_defaults(command=cloudwatch_command)

	from project_pactum.aws.command import list_command
	list_parser = subparsers.add_parser('list', help=None)
	list_parser.set_defaults(command=list_command)

	from project_pactum.aws.command import terminate_command
	terminate_parser = subparsers.add_parser('terminate', help=None)
	terminate_parser.set_defaults(command=terminate_command)
	terminate_parser.add_argument('instance_ids', metavar='instance-id', nargs='+')

def add_command(options):
	from .instance import add_instance
	add_instance()

def cloudwatch_command(options):
	from .cloudwatch import test
	test()

def list_command(options):
	from .instance import list_instances
	list_instances()

def terminate_command(options):
	from .instance import terminate_instances
	terminate_instances(options.instance_ids)

def test_command(options):
	from .instance import get_public_ips
	ips = get_public_ips()
	print('Public IPs')
	for ip in ips:
		print('-', ip)
