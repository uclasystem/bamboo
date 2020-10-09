import boto3

def get_public_ips():
	ips = []

	ec2 = boto3.client('ec2')
	response = ec2.describe_instances()
	for reservation in response['Reservations']:
		for instance in reservation['Instances']:
			if 'SpotInstanceRequestId' not in instance:
				continue
			if instance['ImageId'] != 'ami-0676dd00c188dc297':
				continue
			if 'PublicIpAddress' not in instance:
				continue
			ips.append(instance['PublicIpAddress'])
	return ips
