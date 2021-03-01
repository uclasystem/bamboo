import csv
import datetime
import json
import os
import threading

import project_pactum

from project_pactum.aws.cloudwatch import get_latest_timestamp, get_all_log_events
from project_pactum.aws.instance import create_instance, terminate_instances

class Coordinator:

	def __init__(self, count, instance_type, zone):
		self.lock = threading.Lock()
		self.running = True

		self.cloudwatch_latest_timestamp = get_latest_timestamp()

		self.count = count
		self.instance_type = instance_type
		self.zone = zone
		try:
			instances = create_instance(self.count, self.instance_type, self.zone)
			self.active_servers = [x.id for x in instances]
		except:
			self.active_servers = []

		self.csv_path = os.path.join(project_pactum.BASE_DIR, 'daemon-log.txt')
		self.csv_file = open(self.csv_path, 'a', buffering=1)
		self.csv_writer = csv.writer(self.csv_file)
		self.start_time = datetime.datetime.now()
		self.write_active_servers()

	def write_active_servers(self):
		current_time = datetime.datetime.now()
		delta = current_time - self.start_time
		delta_seconds = delta.days * 86400 + delta.seconds
                # TODO: Remove
		print(len(self.active_servers), 'active servers (writing)')
		self.csv_writer.writerow([delta_seconds, len(self.active_servers), current_time])

	def get_reply(self, msg):
		if msg == 'list':
			return self.list_reply()
		else:
			return 'Unknown'

	def list_reply(self):
		with self.lock:
			return self._list_reply()

	def _list_reply(self):
		if len(self.active_servers) == 0:
			return 'No active servers'
		return '\n'.join(['{} active servers'.format(len(self.active_servers))] + self.active_servers)

	def shutdown(self):
		self.running = False
		self.terminate_all()

	def terminate_all(self):
		with self.lock:
			if len(self.active_servers) == 0:
				return
			terminate_instances(self.active_servers)
			self.active_servers = []

	def is_running(self):
		return self.running

	def ensure_count(self):
		with self.lock:
			remaining = self.count - len(self.active_servers)
		if remaining == 0:
			return
		try:
			instances = create_instance(remaining, self.instance_type, self.zone)
		except:
			instances = []
		if len(instances) == 0:
			return

		with self.lock:
			self.write_active_servers()
			self.active_servers = self.active_servers + [x.id for x in instances]
			self.write_active_servers()

	# TODO: Understand why we are missing some interruption notices
	def check_cloudwatch(self):
		if not self.running:
			return

		interrupted_instance_ids = []
		for event in get_all_log_events(self.cloudwatch_latest_timestamp):
			if event['timestamp'] > self.cloudwatch_latest_timestamp:
				self.cloudwatch_latest_timestamp = event['timestamp']
			message = json.loads(event['message'])
                        # TODO: Handle interruption here
			interrupted_instance_ids.append(message['detail']['instance-id'])

		with self.lock:
			if self.check_server_ids(interrupted_instance_ids):
				self.record_instance_change(interrupted_instance_ids)


	# Backup check to make sure we didn't miss any instances that have been
	# terminated
	def check_terminated(self):
		if not self.running:
			return

                raise RuntimeError('TODO: Refactor with get_instances()')
		finished_instances = []
		# response = describe_instances(self.active_servers)
		for res in response['Reservations']:
			for instance in res['Instances']:
				state = instance['State']['Name']
				if state not in ['running', 'pending']:
					finished_instances.append(instance['InstanceId'])

		with self.lock:
			if self.check_server_ids(finished_instances):
				print("REMOVING TERMINATED INSTANCES. This means some instances were missed")
				self.record_instance_change(finished_instances)

	def check_server_ids(self, server_ids):
		for instance_id in server_ids:
			if instance_id in self.active_servers:
				return True

	def record_instance_change(self, server_ids):
		self.write_active_servers()
		for instance_id in server_ids:
			try:
				self.active_servers.remove(instance_id)
			except ValueError:
				pass
		self.write_active_servers()

