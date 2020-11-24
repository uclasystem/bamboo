import csv
from datetime import datetime
import os
import time

import project_pactum
from project_pactum.aws.instance import create_instance, terminate_instances, InsufficientInstanceCapacity

EXPERIMENT_DIR = os.path.join(project_pactum.BASE_DIR, 'experiment', 'aws-availability')
HISTORY_PATH = os.path.join(EXPERIMENT_DIR, 'history.csv')
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
FIGURES_DIR = os.path.join(project_pactum.BASE_DIR, 'doc', 'figures')

def analyze_daily_merge(old_data, new_data):
	if len(old_data) == 0:
		return new_data

	data = []
	while len(old_data) != 0 or len(new_data) != 0:
		if old_data[0][0] == new_data[0][0]:
			x = old_data.pop(0)
			y = new_data.pop(0)
			assert x[0] == y[0]
			data.append((x[0], x[1] + y[1]))
			previous_new_value = y[1]
		elif old_data[0][0] < new_data[0][0]:
			x = old_data.pop(0)
			y = old_data.pop(0)
			assert x[0] == y[0]
			diff = y[1] - x[1]
			prev = data[-1][1]
			data.append((x[0], prev))
			data.append((x[0], prev + diff))
		else:
			x = new_data.pop(0)
			y = new_data.pop(0)
			assert x[0] == y[0]
			diff = y[1] - x[1]
			prev = data[-1][1]
			data.append((x[0], prev))
			data.append((x[0], prev + diff))
	return data
        
def analyze_daily_from_reader(reader):
	first = True
	data = []
	new_data = []
	dates = []
	available_global_start = None
	available_global_end = None
	available_start = None
	available_durations = []
	for row in reader:
		instance_type = row[1]
		if instance_type != 'p2.xlarge':
			continue

		availability_zone = row[2]
		if availability_zone != 'us-east-1d':
			continue

		d = datetime.fromisoformat(row[0])
		date = d.date().isoformat()
		time_of_day = d.hour * SECONDS_PER_HOUR + d.minute * SECONDS_PER_MINUTE + d.second

		a = row[3] 
		if a == 'allow':
			allowed = True
		elif a == 'deny':
			allowed = False
		elif a == 'unknown':
			allowed = False
		else:
			assert False

		if first:
			previous_date = date
			previous_time_of_day = time_of_day
			previous_allowed = allowed
			first = False
			continue

		if previous_date != date:
			if len(new_data) != 0:
				if previous_allowed:
					new_data.append((24 * SECONDS_PER_HOUR, 1))
				else:
					new_data.append((24 * SECONDS_PER_HOUR, 0))
				dates.append(previous_date)
				data = analyze_daily_merge(data, new_data)
                                
			new_data = []
			if previous_allowed:
				new_data.append((0, 1))
			else:
				new_data.append((0, 0))

		if len(new_data) != 0 and previous_allowed != allowed:
			if previous_allowed:
				new_data.append((time_of_day, 1))
				new_data.append((time_of_day, 0))
				if available_start:
					delta = d - available_start
					assert delta.days == 0
					available_durations.append(delta.seconds)
					available_global_end = d
			else:
				new_data.append((time_of_day, 0))
				new_data.append((time_of_day, 1))
				available_start = d
				if not available_global_start:
					available_global_start = d

		previous_date = date
		previous_time_of_day = time_of_day
		previous_allowed = allowed
		previous_d = d
	from statistics import mean, median, stdev
	print('Mean:', mean(available_durations))
	print('  Standard deviation:', stdev(available_durations))
	print('Median:', median(available_durations))
	print('Min:', min(available_durations))
	print('Max:', max(available_durations))
	total_time_diff =  available_global_end - available_global_start
	print('Uptime: {:.2%}'.format(sum(available_durations) / (total_time_diff.days * SECONDS_PER_DAY + total_time_diff.seconds)))
	return (dates, data)

def analyze_daily_write(dates, data):
	os.makedirs(FIGURES_DIR, exist_ok=True)
	filename = 'aws-availability-{}-to-{}.tex'.format(dates[0], dates[-1])
	with open(os.path.join(FIGURES_DIR, filename), 'w') as f:
		f.write('\\documentclass[crop,convert,tikz]{standalone}\n')
		f.write('\\usetikzlibrary{datavisualization}\n')
		f.write('\\begin{document}\n')
		f.write('\\begin{tikzpicture}\n')
		f.write('  \\datavisualization[\n')
		f.write('    visualize as line=data,\n')
		f.write('    data={\n')
		f.write('      style={line width=0.4pt},\n')
		f.write('    },\n')
		f.write('    scientific axes=clean,\n')
		f.write('    x axis={\n')
		f.write('      label={Hour},\n')
		f.write('      ticks={step=3},\n')
		f.write('    },\n')
		f.write('    y axis={\n')
		f.write('      include value={},\n'.format(len(dates)))
		f.write('      label={Days Available},\n')
		f.write('    },\n')
		f.write('  ]\n')
		f.write('  data {\n')
		f.write('    x, y\n')
		for x, y in data:
			f.write('    {:.3f}, {}\n'.format(x / SECONDS_PER_HOUR, y))
		f.write('  };\n')
		f.write('\\end{tikzpicture}\n')
		f.write('\\end{document}')

def analyze_daily():
	with open(HISTORY_PATH, 'r') as f:
		reader = csv.reader(f)
		dates, data = analyze_daily_from_reader(reader)
		analyze_daily_write(dates, data)

def monitor_iteration(writer, instance_type, available_zones, created=False):
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
			instances = create_instance(1, instance_type, zone)
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

def monitor():
	os.makedirs(EXPERIMENT_DIR, exist_ok=True)
	instance_types = ['p2.16xlarge', 'p2.8xlarge', 'p2.xlarge']
	with open(HISTORY_PATH, 'a', buffering=1) as f:
		writer = csv.writer(f)
		i = 0
		while True:
			i += 1
			print('Iteration', i)
			available_zones = []
			created = False
			for instance_type in instance_types:
				print('  Instance type:', instance_type)
				created = monitor_iteration(writer, instance_type, available_zones, created)
        		# Wait for an hour if we got an instance, otherwise only wait 5 minutes
			if created:
				time.sleep(60 * 60 * 1)
			else:
				time.sleep(60 * 5)
