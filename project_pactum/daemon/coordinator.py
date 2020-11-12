import json

from project_pactum.aws.cloudwatch import get_latest_event, get_events

class Coordinator:

	def __init__(self):
		self.active_servers = []
		event = get_latest_event()
		if event:
			self.cloudwatch_start_time = event['timestamp']
			message = json.loads(event['message'])
			self.cloudwatch_handled_message_ids = [message['id']]
		else:
			self.cloudwatch_start_time = 0
			self.cloudwatch_handled_message_ids = []

	def check_cloudwatch(self):
		cur_handled_message_ids = []
		for event in get_events(self.cloudwatch_start_time):
			if event['timestamp'] > self.cloudwatch_start_time:
				self.cloudwatch_start_time = event['timestamp']
			message = json.loads(event['message'])
			cur_handled_message_ids.append(message['id'])
			if message['id'] in self.cloudwatch_handled_message_ids:
				continue
			print('Interruption warning')
		self.cloudwatch_handled_message_ids = cur_handled_message_ids
