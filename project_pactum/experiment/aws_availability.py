import csv
from datetime import datetime
import os
import time

import project_pactum
from project_pactum.aws.instance import create_instance, terminate_instances, InsufficientInstanceCapacity

def iteration(writer, instance_type, available_zones, created=False):
	# p2_zones = ['us-east-1d', 'us-east-1a', 'us-east-1b', 'us-east-1c', 'us-east-1e']
	# p3_zones = ['us-east-1c', 'us-east-1d', 'us-east-1f']
	zones = ['us-east-1d', 'us-east-1a', 'us-east-1b', 'us-east-1c', 'us-east-1e']
	for zone in zones:
		print('    Testing', zone)

		if zone in available_zones:
			writer.writerow([datetime.now(), instance_type, zone, 'allow'])
			print('      Available zone', zone)
			continue

		if created:
			writer.writerow([datetime.now(), instance_type, zone, 'unknown'])
			print('      Unknown (assume allowed)', zone)
			continue

		try:
			instances = create_instance(zone, instance_type)
			created = True
			writer.writerow([datetime.now(), instance_type, zone, 'allow'])
			for instance in instances:
				print('      Created', zone, instance.id)
			time.sleep(30)
			instance_ids = [x.id for x in instances]
			terminate_instances(instance_ids)
			print('      Terminated', zone, ','.join(instance_ids))
		except InsufficientInstanceCapacity as e:
			available_zones.extend(e.available_zones)
			writer.writerow([datetime.now(), instance_type, zone, 'deny'])
			print('      InsufficientInstanceCapacity', zone, e.message)

	return created

def run():
	experiment_dir = os.path.join(project_pactum.BASE_DIR, 'experiment', 'aws-availability')
	os.makedirs(experiment_dir, exist_ok=True)
	path = os.path.join(experiment_dir, 'history.csv')
	instance_types = ['p2.16xlarge', 'p2.8xlarge', 'p2.xlarge']
	with open(path, 'a', buffering=1) as f:
		writer = csv.writer(f)
		i = 0
		while True:
			i += 1
			print('Iteration', i)
			available_zones = []
			created = False
			for instance_type in instance_types:
				print('  Instance type:', instance_type)
				created = iteration(writer, instance_type, available_zones, created)
        		# Wait for an hour if we got an instance, otherwise only wait 5 minutes
			if created:
				time.sleep(60 * 60 * 1)
			else:
				time.sleep(60 * 5)
