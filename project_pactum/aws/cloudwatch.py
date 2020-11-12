import boto3

LOG_GROUP_NAME = '/aws/events/project-pactum'
LOG_STREAM_NAME = '8f7e040e-0ef6-359d-978c-01ed095f40ae'

def get_latest_event():
	client = boto3.client('logs')
	log_events = client.get_log_events(
		logGroupName=LOG_GROUP_NAME,
		logStreamName=LOG_STREAM_NAME,
		startFromHead=False,
		limit=1,
	)
	if len(log_events['events']) == 0:
		return None
	return log_events['events'][0]

def get_events(start_time):
	client = boto3.client('logs')
	next_forward_token = None
	log_events = client.get_log_events(
		logGroupName=LOG_GROUP_NAME,
		logStreamName=LOG_STREAM_NAME,
		startFromHead=True,
		startTime=start_time,
	)
	while next_forward_token != log_events['nextForwardToken']:
		for event in log_events['events']:
			yield event

		next_forward_token = log_events['nextForwardToken']
		log_events = client.get_log_events(
			logGroupName=LOG_GROUP_NAME,
			logStreamName=LOG_STREAM_NAME,
			startFromHead=True,
			startTime=start_time,
			nextToken=next_forward_token,
		)
