## Imports
import os

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse

import trace_parser as tp

import boto3


idle_time_parser = argparse.ArgumentParser()
idle_time_parser.add_argument('--start', '-s', type=str, default=None)
idle_time_parser.add_argument('--end', '-e', type=str, default=None)
idle_time_parser.add_argument('--period', '-p', type=int, default=60)
idle_time_parser.add_argument('--stat-type', type=str, default='Maximum')
idle_time_parser.add_argument('--auto-scaling-group', '-asg', type=str, default='pipeline-mngr-test')
idle_time_parser.add_argument('--force', action='store_true')

idle_time_parser.add_argument('--base-dir', '-d', type=str, default=None)

args = idle_time_parser.parse_args()

## Global Vars
BASE_DIR=args.base_dir
os.chdir(BASE_DIR)

dt_format = '%Y %m %d %H %M %S'
start = f'{args.start} 00'
end = f'{args.end} 00'
run_start_dt = datetime.strptime(start, dt_format)
run_end_dt = datetime.strptime(end, dt_format)

start_str = run_start_dt.strftime('%Y-%m-%d_%H-%M')
end_str = run_end_dt.strftime('%Y-%m-%d_%H-%M')

if not os.path.exists(f'trace--{args.auto_scaling_group}--s{start_str}--e{end_str}') or args.force:
    tp.get_parsed_trace(run_start_dt, run_end_dt, args.period, args.stat_type, args.auto_scaling_group)


def dt_to_ts(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def time_str_to_dt(time_str, format="%Y-%m-%d %H:%M:%S"):
    return datetime.strptime(time_str, format)

def convert_to_utc(dt):
    return dt + timedelta(hours=7)

def round_dt(dt):
    if dt.second >= 30:
        dt = dt + timedelta(minutes=1)

    return dt.replace(second=0, microsecond=0)

def dt_from_ts(ts):
    dt = datetime.fromtimestamp(ts)
    dt = convert_to_utc(dt)
    return round_dt(dt)

def filter_range(start, end, dt, column='dt'):
    if start == end:
        end = end + timedelta(minutes=1)
    dt = dt[dt[column] >= start]
    dt = dt[dt[column] <= end]
    return dt

def find_next(dt, df, column='dt_start'):
    tmp = df[df[column] >= dt].iloc[0]
    return tmp['dt_start']

async_chkpts = []
engine_times = []
load_times = []
loop_times = []
for i in range(8):
    async_chkpts.append(pd.read_csv(f'async-checkpoint_{i}.csv', names=['elapsed']))
    engine_times.append(pd.read_csv(f'engine-timing_{i}.csv', names=['op', 'start', 'end', 'elapsed']))
    load_times.append(pd.read_csv(f'load_times_{i}.csv', names=['ws', 'step', 'elapsed']))
    loop_times.append(pd.read_csv(f'loop_times_{i}.csv', names=['ws', 'step', 'start', 'end', 'train', 'chkpt', 'total']))

reconf = pd.read_csv('reconfig.csv', names=['ws', 'start', 'end', 'elapsed'])
reconf['dt_start'] = reconf['start'].apply(lambda x: dt_from_ts(x))
reconf['dt_end'] = reconf['end'].apply(lambda x: dt_from_ts(x))


train_loss_time = pd.read_csv('train-loss-time.csv', names=['timestamp', 'step', 'loss'])
train_loss_time['dt'] = train_loss_time['timestamp'].apply(lambda x: dt_from_ts(x))

TRACE_DIR=f'trace--pipeline-mngr-test--s{run_start_dt.strftime("%Y-%m-%d_%H-%M")}--e{run_end_dt.strftime("%Y-%m-%d_%H-%M")}'
in_service = pd.read_csv(os.path.join(TRACE_DIR, 'pipeline-mngr-test-GroupInServiceInstances.csv'))
terminating = pd.read_csv(os.path.join(TRACE_DIR, 'pipeline-mngr-test-GroupTerminatingInstances.csv'))
pending = pd.read_csv(os.path.join(TRACE_DIR, 'pipeline-mngr-test-GroupPendingInstances.csv'))
total = pd.read_csv(os.path.join(TRACE_DIR, 'pipeline-mngr-test-GroupTotalInstances.csv'))

in_service['dt'] = in_service['time'].apply(lambda x: time_str_to_dt(x))
terminating['dt'] = terminating['time'].apply(lambda x: time_str_to_dt(x))
pending['dt'] = pending['time'].apply(lambda x: time_str_to_dt(x))
total['dt'] = total['time'].apply(lambda x: time_str_to_dt(x))

reconf.to_csv('dt_reconfig.csv')
train_loss_time.to_csv('dt_tlt.csv')

reconfig_ranges = []
running_ranges = []

for t in range(len(reconf)):
    row = reconf.iloc[t]
    reconfig_ranges.append(filter_range(row['dt_start'], row['dt_end'], in_service))
    
    if t < len(reconf) - 1:
        next_row = reconf.iloc[t+1]
        running_ranges.append(filter_range(row['dt_end'], next_row['dt_start'], in_service))
        
last_start = reconf.iloc[len(reconf)-1]['dt_end']
end_end = in_service.iloc[len(in_service)-1]['dt']
running_ranges.append(filter_range(last_start, end_end, in_service))

run_start = reconf.iloc[0]['dt_start']
run_end = train_loss_time.iloc[len(train_loss_time)-1]['dt']
running_range = filter_range(run_start, run_end, in_service)

prev_step = -1
prev_row = None
step = -1
wasted_work_ranges = []
for i in range(len(train_loss_time)):
    row = train_loss_time.iloc[i]
    step = row['step']
    if prev_step >= step:
        ri = i
        tgt_step = step
        curr_step = prev_step
        end = prev_row['dt']
        while curr_step >= tgt_step:
            ri -= 1
            curr_row = train_loss_time.iloc[ri]
            curr_step = curr_row['step']
        
        start_row = train_loss_time.iloc[ri]
        start = start_row['dt']
        end = find_next(end, reconf)
        wasted_work_ranges.append(filter_range(start, end, in_service))
    
    prev_row = row
    prev_step = step

fig, ax = plt.subplots(figsize=(8, 6))
date_format = mdates.DateFormatter("%m-%d %H:%M")
ax.plot(in_service['dt'], in_service['count'])

ax.fill_between(running_range['dt'], running_range['count'])

for r in wasted_work_ranges:
    ax.fill_between(r['dt'], r['count'], color='orange')
    
for r in reconfig_ranges:
    ax.fill_between(r['dt'], r['count'], color='red')

ax.grid()
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
graph_fname = f'idle-time--s{run_start_dt.strftime("%Y-%m-%d_%H-%M")}--e{run_end_dt.strftime("%Y-%m-%d_%H-%M")}.png'
plt.savefig(graph_fname, format='png', dpi=240, bbox_inches='tight')