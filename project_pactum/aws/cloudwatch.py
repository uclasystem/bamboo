import boto3

LOG_GROUP_NAME = '/aws/events/project-pactum'

def get_latest_event_time():
	client = boto3.client('logs')
	response = client.describe_log_streams(
		logGroupName=LOG_GROUP_NAME,
		orderBy='LastEventTime',
		descending=True,
	)
	try:
		return response['logStreams'][0]['lastEventTimestamp']
	except:
		return 0

def get_events(latest_event_time):
	client = boto3.client('logs')
	response = client.describe_log_streams(
		logGroupName=LOG_GROUP_NAME,
		orderBy='LastEventTime',
		descending=True,
	)
	while True:
		done = False
		for log_stream in response['logStreams']:
			if log_stream['lastEventTimestamp'] <= latest_event_time:
				done = True
				break
			for event in get_events_for_stream(log_stream['logStreamName'], latest_event_time):
				yield event
		if done:
			break
		if not 'nextToken' in response:
			break
		response = client.describe_log_streams(
			logGroupName=LOG_GROUP_NAME,
			orderBy='LastEventTime',
			descending=True,
			nextToken=response['nextToken'],
		)

def get_events_for_stream(log_stream_name, latest_event_time):
	client = boto3.client('logs')
	next_forward_token = None
	log_events = client.get_log_events(
		logGroupName=LOG_GROUP_NAME,
		logStreamName=log_stream_name,
		startFromHead=True,
		startTime=latest_event_time,
	)
	while next_forward_token != log_events['nextForwardToken']:
		for event in log_events['events']:
			if event['timestamp'] > latest_event_time:
				yield event

		next_forward_token = log_events['nextForwardToken']
		log_events = client.get_log_events(
			logGroupName=LOG_GROUP_NAME,
			logStreamName=log_stream_name,
			startFromHead=True,
			startTime=latest_event_time,
			nextToken=next_forward_token,
		)
