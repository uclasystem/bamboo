import boto3

import re

IMAGE_ID = 'ami-09696628562d9969d'

def describe_instances(instance_ids):
	ec2 = boto3.client('ec2')
	return ec2.describe_instances(instance_ids)

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

# An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation (reached max retries: 4): We currently do not have sufficient p2.16xlarge capacity in the Availability Zone you requested (us-east-1d). Our system will be working on provisioning additional capacity. You can currently get p2.16xlarge capacity by not specifying an Availability Zone in your request or choosing us-east-1a, us-east-1b, us-east-1c, us-east-1e.

# An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation (reached max retries: 4): There is no Spot capacity available that matches your request.

class InsufficientInstanceCapacity(Exception):

	def __init__(self, available_zones, message):
		self.available_zones = available_zones
		self.message = message

def parse_run_instances_exception(s):
	m = re.match(r'An error occurred \(([^)]+)\) when calling the RunInstances operation \(reached max retries: (\d+)\): ', s)
	assert m.group(1) == 'InsufficientInstanceCapacity'
	message = s[len(m.group(0)):]

	m = re.match(r'We currently do not have sufficient ([a-z0-9.]+) capacity in the Availability Zone you requested \(([a-z0-9-]+)\). Our system will be working on provisioning additional capacity. You can currently get ([a-z0-9.]+) capacity by not specifying an Availability Zone in your request or choosing ([^.]+)\.', message)
	if not m:
		raise InsufficientInstanceCapacity([], message)
	assert m.group(1) == m.group(3)
	available_zones = m.group(4).split(', ')
	raise InsufficientInstanceCapacity(available_zones, message)

def create_instance(num_instances, instance_type, availability_zone=None,
					image_id=IMAGE_ID):
	ec2 = boto3.resource('ec2')
	instances = []
	args = {
		'ImageId': image_id,
		'InstanceType': instance_type,
		'MinCount': 1,
		'MaxCount': num_instances,
		'InstanceMarketOptions': {
			'MarketType': 'spot',
			'SpotOptions': {
				'SpotInstanceType': 'one-time',
				'InstanceInterruptionBehavior': 'terminate',
			}
		},
		'SecurityGroupIds': ['sg-0a7c9ebe69b2b770b'],
		'EbsOptimized': True
	}

	# Only specify the availability zone if specified in args
	if availability_zone != None:
		args['Placement'] = { 'AvailabilityZone': availability_zone }

	try:
		response = ec2.create_instances(**args)
		instances = response
	except Exception as e:
		parse_run_instances_exception(str(e))
	return instances

def add_instance():
	ec2 = boto3.resource('ec2')
	instances = create_instance(1, 'p2.xlarge', 'us-east-1d')
	for instance in instances:
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
