import boto3

LOG_GROUP_NAME = '/aws/events/project-pactum'

def get_log_streams(limit=None):
	kwargs = {
	        'logGroupName': LOG_GROUP_NAME,
		'orderBy': 'LastEventTime',
		'descending': True,
	}
	if limit:
		kwargs['limit'] = limit
	client = boto3.client('logs')
	response = client.describe_log_streams(**kwargs)
	while True:
		for log_stream in response['logStreams']:
			yield log_stream
		if not 'nextToken' in response:
			break
		kwargs['nextToken'] = response['nextToken']
		response = client.describe_log_streams(**kwargs)

def get_log_events(log_stream_name, start_time=None):
	kwargs = {
	        'logGroupName': LOG_GROUP_NAME,
		'logStreamName': log_stream_name,
		'startFromHead': False,
	}
	if start_time:
		kwargs['startTime'] = start_time
	client = boto3.client('logs')
	next_backward_token = None
	log_events = client.get_log_events(**kwargs)
	while next_backward_token != log_events['nextBackwardToken']:
		# Tested with limit=1 argument for get_log_events
		for event in reversed(log_events['events']):
			yield event
		next_backward_token = log_events['nextBackwardToken']
		kwargs['nextToken'] = next_backward_token
		log_events = client.get_log_events(**kwargs)

def delete_log_streams():
	client = boto3.client('logs')
	for log_stream_name in get_log_stream_names():
		response = client.delete_log_stream(
			logGroupName=LOG_GROUP_NAME,
			logStreamName=log_stream_name
		)

def get_log_stream_names():
	for log_stream in get_log_streams():
		yield log_stream['logStreamName']

def get_latest_timestamp():
	for log_stream in get_log_streams(limit=1):
		return log_stream['lastEventTimestamp']
	return 0

def get_all_log_events(start_time):
	for log_stream in get_log_streams():
		if log_stream['lastEventTimestamp'] <= start_time:
			break
		log_stream_name = log_stream['logStreamName']
		for event in get_log_events(log_stream_name, start_time=start_time):
			if event['timestamp'] <= start_time:
				break
			yield event
