import boto3

def test_command(options):
	ec2 = boto3.client('ec2')
	response = ec2.describe_instances()
	for reservation in response['Reservations']:
		for instance in reservation['Instances']:
			if 'InstanceLifeCycle' in instance and instance['InstanceLifeCycle'] != 'spot':
				continue
			public_ip_address = 'OFFLINE'
			if 'PublicIpAddress' in instance:
				public_ip_address = instance['PublicIpAddress']
			print(instance['InstanceId'], 'spot', public_ip_address)
