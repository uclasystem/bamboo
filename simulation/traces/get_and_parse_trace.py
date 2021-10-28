import argparse
from datetime import datetime, timedelta
import time
import os
import json
import boto3


ONE_REQUEST_MAX=1440

metrics_request_parser = argparse.ArgumentParser()
metrics_request_parser.add_argument('--start', '-s', type=str, default=None)
metrics_request_parser.add_argument('--end', '-e', type=str, default=None)
metrics_request_parser.add_argument('--period', '-p', type=int, default=60)
metrics_request_parser.add_argument('--stat-type', type=str, default='Maximum')
metrics_request_parser.add_argument('--auto-scaling-group', '-asg', type=str, default='pipeline-mngr-test')
metrics_request_parser.add_argument('--region', '-r', type=str, default='us-east-1')

args = metrics_request_parser.parse_args()

cw_client = boto3.client('cloudwatch', region_name=args.region)

dt_format = '%Y %m %d %H %M %S'
args.start = f'{args.start} 00'
args.end = f'{args.end} 00'
start_dt = datetime.strptime(args.start, dt_format)
end_dt = datetime.strptime(args.end, dt_format)

curr_upper_bound = end_dt
curr_lower_bound = max(end_dt - timedelta(seconds=(args.period * ONE_REQUEST_MAX)), start_dt)

metric_data_points = {
		'GroupTotalInstances': [],
		'GroupTerminatingInstances': [],
		'GroupPendingInstances': [],
		'GroupInServiceInstances': []
	}

start_str = start_dt.strftime('%Y-%m-%d_%H-%M')
end_str = end_dt.strftime('%Y-%m-%d_%H-%M')
results_dir = f'trace--{args.auto_scaling_group}--s{start_str}--e{end_str}'
os.makedirs(results_dir)

while True:
	for metric in metric_data_points:
		response = cw_client.get_metric_statistics(
			Namespace='AWS/AutoScaling',
			MetricName=metric,
			Dimensions = [
			{
				'Name': 'AutoScalingGroupName',
				'Value': args.auto_scaling_group
			}],
			StartTime=curr_lower_bound,
			EndTime=curr_upper_bound,
			Period=args.period,
			Statistics=[
				args.stat_type,
			]
		)

		metric_data_points[metric].extend(response['Datapoints'])


	if curr_lower_bound <= start_dt:
		break

	# Sleep just in case AWS API rate limiter kicks in
	time.sleep(1)

	curr_upper_bound = curr_lower_bound
	curr_lower_bound = curr_lower_bound - timedelta(seconds=(args.period * ONE_REQUEST_MAX))


for metric in metric_data_points:
	metric_data_points[metric].sort(key=lambda x : x['Timestamp'])

	with open(os.path.join(results_dir, f'{args.auto_scaling_group}-{metric}.csv'), 'w') as f:
		f.write('time,count\n')
		for dp in metric_data_points[metric]:
			timestamp = dp['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
			cnt = dp[args.stat_type]
			f.write(f'{timestamp},{cnt}\n')
