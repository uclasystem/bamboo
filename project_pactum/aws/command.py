from .instance import get_public_ips

def test_command(options):
	ips = get_public_ips()
	print('Public IPs')
	for ip in ips:
		print('-', ip)
