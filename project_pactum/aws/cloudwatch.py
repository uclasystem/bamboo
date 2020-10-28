import boto3

def test():
	client = boto3.client('cloudwatch')
	print(client.describe_alarms())
