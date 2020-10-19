def add_command(options):
	from .instance import add_instance
	add_instance()

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
