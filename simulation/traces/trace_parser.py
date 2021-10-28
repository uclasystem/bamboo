import argparse
from datetime import datetime, timedelta
import time
import os
import json
import boto3


ONE_REQUEST_MAX=1440
cw_client = boto3.client('cloudwatch')


def get_parsed_trace(start_dt, end_dt, period, stat_type, auto_scaling_group):
	curr_upper_bound = end_dt
	curr_lower_bound = max(end_dt - timedelta(seconds=(period * ONE_REQUEST_MAX)), start_dt)

	metric_data_points = {
			'GroupTotalInstances': [],
			'GroupTerminatingInstances': [],
			'GroupPendingInstances': [],
			'GroupInServiceInstances': []
		}

	start_str = start_dt.strftime('%Y-%m-%d_%H-%M')
	end_str = end_dt.strftime('%Y-%m-%d_%H-%M')
	results_dir = f'trace--{auto_scaling_group}--s{start_str}--e{end_str}'
	os.makedirs(results_dir)

	while True:
		for metric in metric_data_points:
			response = cw_client.get_metric_statistics(
				Namespace='AWS/AutoScaling',
				MetricName=metric,
				Dimensions = [
				{
					'Name': 'AutoScalingGroupName',
					'Value': auto_scaling_group
				}],
				StartTime=curr_lower_bound,
				EndTime=curr_upper_bound,
				Period=period,
				Statistics=[
					stat_type,
				]
			)

			metric_data_points[metric].extend(response['Datapoints'])


		if curr_lower_bound <= start_dt:
			break

		# Sleep just in case AWS API rate limiter kicks in
		time.sleep(1)

		curr_upper_bound = curr_lower_bound
		curr_lower_bound = curr_lower_bound - timedelta(seconds=(period * ONE_REQUEST_MAX))

	for metric in metric_data_points:
		metric_data_points[metric].sort(key=lambda x : x['Timestamp'])

		with open(os.path.join(results_dir, f'{auto_scaling_group}-{metric}.csv'), 'w') as f:
			f.write('time,count\n')
			for dp in metric_data_points[metric]:
				timestamp = dp['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
				cnt = dp[stat_type]
				f.write(f'{timestamp},{cnt}\n')
