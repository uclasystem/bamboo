import csv
import datetime
import json
import os
import threading

import project_pactum

from project_pactum.aws.cloudwatch import get_latest_event_time, get_events
from project_pactum.aws.instance import create_instance, terminate_instances

class Coordinator:

	def __init__(self, count, instance_type, zone):
		self.lock = threading.Lock()
		self.running = True

		self.cloudwatch_latest_event_time = get_latest_event_time()

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
		self.csv_writer.writerow([delta_seconds, len(self.active_servers)])

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
			self.active_servers = self.active_servers + [x.id for x in instances]
			self.write_active_servers()

	def check_cloudwatch(self):
		if not self.running:
			return

		interrupted_instance_ids = []
		for event in get_events(self.cloudwatch_latest_event_time):
			if event['timestamp'] > self.cloudwatch_latest_event_time:
				self.cloudwatch_latest_event_time = event['timestamp']
			message = json.loads(event['message'])
                        # TODO: Handle interruption here
			interrupted_instance_ids.append(message['detail']['instance-id'])

		with self.lock:
			removed_servers = False
			for instance_id in interrupted_instance_ids:
				print('Trying to remove {} (interrupted)'.format(instance_id))
				try:
					self.active_servers.remove(instance_id)
					print('  Removed {} (interrupted)'.format(instance_id))
					removed_servers = True
				except ValueError:
					pass
			if removed_servers:
				self.write_active_servers()
