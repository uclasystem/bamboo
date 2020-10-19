import boto3

IMAGE_ID = 'ami-0660425fc55822178'

def get_instances():
	instances = []

	ec2 = boto3.client('ec2')
	response = ec2.describe_instances()
	for reservation in response['Reservations']:
		for instance in reservation['Instances']:
			if 'SpotInstanceRequestId' not in instance:
				continue
			if instance['ImageId'] != IMAGE_ID:
				continue
			if 'PublicIpAddress' not in instance:
				continue
			instances.append(instance)
	return instances

def get_private_ips():
	return [x['PrivateIpAddress'] for x in get_instances()]

def get_public_ips():
	return [x['PublicIpAddress'] for x in get_instances()]

def get_instance_ids():
	return [x['InstanceId'] for x in get_instances()]

def add_instance():
	ec2 = boto3.resource('ec2')
	response = ec2.create_instances(
		ImageId=IMAGE_ID,
		InstanceType='t2.micro',
		MinCount=1,
		MaxCount=1,
		InstanceMarketOptions={
			'MarketType': 'spot',
			'SpotOptions': {
				'SpotInstanceType': 'one-time',
				'InstanceInterruptionBehavior': 'terminate',
			}
		},
		SecurityGroupIds=['sg-0a7c9ebe69b2b770b'],
	)
	for instance in response:
		print(instance.id)

def list_instances():
	ec2 = boto3.resource('ec2')
	for instance in ec2.instances.all():
		print(instance.id)
	# instances = get_instances()
	# for instance_id in get_instance_ids():
	# 	print(instance_id)

def terminate_instances(instance_ids):
	ec2 = boto3.resource('ec2')
	ec2.instances.filter(InstanceIds=instance_ids).terminate()
