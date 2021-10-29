#!/usr/bin/env python
# coding: utf-8

# #  Simulation Framework
# ## Idea is to estimate different levels of idle times at different levels of preemption

import os
import random
import glob
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta

BASE_DIR=os.path.join('profiling-runs', 'run-86')

parser = argparse.ArgumentParser()
parser.add_argument('--preemption-chance-per-minute', '-pcm', type=float, default=0.25)
parser.add_argument('--preemption-distribution', '-pd',
    type=str, default='{"mean": 0, "std": 1}')
parser.add_argument('--addition-chance-per-minute', '-acm', type=float, default=0.25)
parser.add_argument('--addition-distribution', '-ad',
    type=str, default='{"mean": 0, "std": 1}')
parser.add_argument('--total-minutes', '-tm', type=int, default=None)

args = parser.parse_args()

total_minutes = args.total_minutes
args.preemption_distribution = json.loads(args.preemption_distribution)
args.addition_distribution = json.loads(args.addition_distribution)

def ts_to_dt(ts, format="%Y-%m-%dT%H:%M:%SZ"):
    return datetime.strptime(ts, format)

def round_values(d):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = round_values(v)
        else:
            new_dict[k] = np.round(v)
    
    return new_dict

def get_next_dir():
    base_dir = 'simulation_results'
    os.makedirs(base_dir, exist_ok=True)

    existing_trials_so_far = glob.glob(os.path.join(base_dir, 'run-*'))
    existing_trials_so_far = [int(r[r.rfind('-')+1:]) for r in existing_trials_so_far]
    next_run = 0 if len(existing_trials_so_far) == 0 else max(existing_trials_so_far) + 1

    return os.path.join(base_dir, f'run-{next_run}')

## Set total minutes based on the real trace if none is provided
if total_minutes == None:
    node_trace = pd.read_csv(os.path.join('traces', 'trace--full-64-node-trace', 'full-64-node-trace.csv'))
    node_trace['dt'] = node_trace['timestamp'].apply(lambda x: ts_to_dt(x))
    trace_start = node_trace.iloc[0]['dt']
    trace_end = node_trace.iloc[len(node_trace)-1]['dt']
    trace_timedelta = trace_end - trace_start
    total_minutes = trace_timedelta.total_seconds() / 60

## Fill out performance stats based on profiling
performance_stats = {}

reconfig = pd.read_csv(os.path.join(BASE_DIR, 'reconfig.csv'), names=['ws', 'start', 'end', 'elapsed'])
reconfig = reconfig.drop([0])
mean_reconfig_time = reconfig['elapsed'].mean()
performance_stats['reconfiguration'] = mean_reconfig_time


loop_times = pd.read_csv(os.path.join(BASE_DIR, 'loop_times_0.csv'), names=['ws', 'step', 'loop start', 'loop end', 'train time', 'start chkpt thread', 'total loop time'])
ws_iter_times = {}
for ws in loop_times.ws.unique():
    filtered_df = loop_times[loop_times['ws'] == ws]
    ws_iter_times[ws] = filtered_df['train time'].mean()

performance_stats['iter_times'] = ws_iter_times


chkpt_times = pd.read_csv(os.path.join(BASE_DIR, 'async-checkpoint_0.csv'), names=['elapsed'])
mean_chkpt_time = chkpt_times['elapsed'].mean()
performance_stats['checkpoint'] = mean_chkpt_time


load_times = pd.read_csv(os.path.join(BASE_DIR, 'load_times_0.csv'), names=['ws', 'step', 'elapsed'])
mean_load_times = load_times['elapsed'].mean()
performance_stats['load checkpoint'] = mean_load_times
performance_stats = round_values(performance_stats)

print(performance_stats)


def run_simulation(preemption_chance_per_minute, preemption_distribution, addition_chance_per_minute, addition_distribution, simulation_length_minutes, exp_dir):
    start = datetime(2021, 9, 14, 17, 00)
    end = start + timedelta(minutes=simulation_length_minutes)

    active_instances = set()
    standy_instances = set()
    prepping_instances = set()
    all_instances = set()

    prepping_timers = {}
    current_state = 'training'

    preemption_chance = preemption_chance_per_minute / 12
    addition_chance = addition_chance_per_minute / 12

    current_instances = 32
    pipeline_depth = 16
    id = 0
    for _ in range(current_instances):
        all_instances.add(id)
        active_instances.add(id)
        id += 1

    step = 0
    trace = {
        'ws': [current_instances],
        'dt': [start],
        'event': ['step'],
        'event-id': [0]
    }

    reconfig_cnt = 0
    reconfig_end_time = None
    last_saved_iteration = 0
    checkpoint_running = False
    checkpoint_start = None
    checkpoint_step = 0

    inst_min = 0
    inst_max = 64
    curr_time = start
    f = open(os.path.join(exp_dir, 'simulation-output.txt'), 'w')
    instance_count_trace = open(os.path.join(exp_dir, 'in-service-trace.csv'), 'w')
    instance_count_trace.write('timestamp,count\n')
    try:
        while curr_time < end:
            instance_count_trace.write(f'{curr_time},{len(all_instances)}\n')

            event_str = '' if current_state == 'training' else f'reconfig end {reconfig_end_time}'
            state_msg = f'Next event: {event_str}'
            f.write(f'{curr_time.strftime("%Y-%m-%d %H:%M:%S")} Cluster Config this step:\n')
            f.write(f'######## Current status: {current_state}; {state_msg}\n')
            f.write(f'######## {len(active_instances)} active instances: {active_instances}\n')
            f.write(f'######## {len(standy_instances)} standby instances: {standy_instances}\n')
            f.write(f'######## {len(prepping_instances)} prepping instances: {prepping_instances}\n')
            f.write(f'######## {len(all_instances)} total instances: {all_instances}\n')
            f.write('========================================================================\n')

            num_to_add = 0
            add_this_round = np.random.random() < addition_chance
            if add_this_round:
                big_or_regular = np.random.random()
                if big_or_regular < 0.058:
                    num_to_add = np.random.randint(11, 32)
                    if len(all_instances) + num_to_add >= inst_max:
                        num_to_add = inst_max - len(all_instances)
                else:
                    num_to_add = np.random.normal(addition_distribution['mean'], addition_distribution['std'])
                    num_to_add = np.abs(np.round(num_to_add))
                    if num_to_add == 0:
                        num_to_add = 1
                    elif len(all_instances) + num_to_add >= inst_max:
                        num_to_add = inst_max - len(all_instances)

            num_to_preempt = 0
            preempt_this_round = np.random.random() < preemption_chance
            if preempt_this_round and not len(all_instances) <= inst_min:
                num_to_preempt = np.random.normal(preemption_distribution['mean'], preemption_distribution['std'])
                num_to_preempt = np.abs(np.round(num_to_preempt))
                if num_to_preempt == 0:
                    num_to_preempt = 1

            preempted_instances = random.sample(all_instances, int(num_to_preempt))
            lost_active = False
            preempted_active = []
            preempted_stdby = []
            preempted_prep = []
            for inst_id in preempted_instances:
                if inst_id in active_instances:
                    lost_active = True
                    active_instances.remove(inst_id)
                    preempted_active.append(inst_id)
                elif inst_id in standy_instances:
                    standy_instances.remove(inst_id)
                    preempted_stdby.append(inst_id)
                elif inst_id in prepping_instances:
                    prepping_instances.remove(inst_id)
                    del prepping_timers[inst_id]
                    preempted_prep.append(inst_id)
                else:
                    print(f'What the F?')

                all_instances.remove(inst_id)

            msg = f'Preempting {len(preempted_active)} active, {len(preempted_stdby)} stdby, {len(preempted_prep)} prep\n'
            msg += f'Active lost: {preempted_active}, lost stdby: {preempted_stdby}, lost prep {preempted_prep}'
            if len(preempted_instances) > 0:
                f.write(f'{msg}\n')


            done_prepping = []
            for inst_id in prepping_instances:
                time_spent_prepping = curr_time - prepping_timers[inst_id]
                if time_spent_prepping.total_seconds() >= 180:
                    done_prepping.append(inst_id)

            if len(done_prepping) > 0:
                f.write(f'Moving {len(done_prepping)} instances to standby with ids {done_prepping}\n')
            for inst_id in done_prepping:
                standy_instances.add(inst_id)
                prepping_instances.remove(inst_id)
                del prepping_timers[inst_id]

            if num_to_add > 0:
                f.write(f'Adding {num_to_add} new instances with sart id {id}\n')
            for _ in range(int(num_to_add)):
                all_instances.add(id)
                prepping_instances.add(id)
                prepping_timers[id] = curr_time
                id += 1

            new_pipeline_avail = len(standy_instances) >= pipeline_depth

            last_event = {
                'ws': trace['ws'][-1],
                'dt': trace['dt'][-1],
                'event': trace['event'][-1],
                'event-id': trace['event-id'][-1]
            }
            if checkpoint_running:
                checkpoint_end_time = checkpoint_start + timedelta(seconds=performance_stats['checkpoint'])
                if curr_time >= checkpoint_end_time:
                    f.write(f'Finished checkpoint {checkpoint_step}\n')
                    last_saved_iteration = checkpoint_step
                    checkpoint_running = False

            if last_event['event'] == 'step':
                iter_end_time = last_event['dt'] + timedelta(seconds=performance_stats['iter_times'][last_event['ws']])
                if curr_time >= iter_end_time:
                    f.write('Finished step {} at {}. Next iter finishes at {}\n'.format(step, iter_end_time, iter_end_time + timedelta(seconds=performance_stats['iter_times'][last_event['ws']])))
                    if not checkpoint_running:
                        checkpoint_start = iter_end_time
                        checkpoint_step = step
                        checkpoint_running = True
                        f.write(f'Starting to write checkpoint {checkpoint_step}\n')

                    step += 1
                    trace['ws'].append(last_event['ws'])
                    trace['dt'].append(iter_end_time)
                    trace['event'].append('step')
                    trace['event-id'].append(step)

            elif last_event['event'] == 'reconfig':
                if lost_active or new_pipeline_avail:
                    msg = f'Extending reconf end time: prev {reconfig_end_time},'
                    reconfig_end_time = curr_time + timedelta(seconds=performance_stats['reconfiguration'])
                    msg += f' new {reconfig_end_time} because {lost_active} {new_pipeline_avail}'
                    f.write(f'{msg}\n')

                if curr_time >= reconfig_end_time:
                    f.write(f'Reconfig done. Training starts: {curr_time} >= {reconfig_end_time}\n')
                    current_state = 'training'
                    step = last_saved_iteration + 1
                    trace['ws'].append(len(active_instances))
                    trace['dt'].append(reconfig_end_time)
                    trace['event'].append('step')
                    trace['event-id'].append(step)
                    reconfig_end_time = None
            else:
                print('huh?')

            if lost_active or new_pipeline_avail:
                current_state = 'reconfiguring'
                trace['ws'].append(len(active_instances))
                trace['dt'].append(curr_time)
                trace['event'].append('reconfig')
                trace['event-id'].append(reconfig_cnt)
                reconfig_cnt += 1
                if reconfig_end_time == None:
                    reconfig_end_time = curr_time + timedelta(seconds=performance_stats['reconfiguration'])
                    f.write(f'Setting the reconfig end time to {reconfig_end_time}\n')

                checkpoint_running = False

                if lost_active and not new_pipeline_avail:
                    f.write('Lost active instance but no new pipeline avail: ')
                    diff = len(active_instances) % pipeline_depth
                    if len(standy_instances) >= pipeline_depth - diff:
                        f.write('case 1a\n')
                        n_to_move = pipeline_depth - diff
                        to_move = random.sample(standy_instances, n_to_move)
                        for k in to_move:
                            active_instances.add(k)
                            standy_instances.remove(k)
                    else:
                        f.write('case 1b\n')
                        to_move = random.sample(active_instances, diff)
                        for k in to_move:
                            standy_instances.add(k)
                            active_instances.remove(k)

                elif not lost_active and new_pipeline_avail:
                    f.write('No instance lost but new pipeline is availalbe\n')
                    n_to_move = (len(standy_instances) // pipeline_depth) * pipeline_depth
                    f.write(f'LEN STDBY: {len(standy_instances)}, pipe depth: {pipeline_depth}, N_TO_MOVE: {n_to_move}\n')
                    to_move = random.sample(standy_instances, n_to_move)
                    for k in to_move:
                        active_instances.add(k)
                        standy_instances.remove(k)

                else:
                    f.write('Both lost an instance AND got a new pipeline\n')
                    diff = len(active_instances) % pipeline_depth
                    n_to_move = ((len(standy_instances) // pipeline_depth) * pipeline_depth) - diff
                    to_move = random.sample(standy_instances, n_to_move)
                    for k in to_move:
                        active_instances.add(k)
                        standy_instances.remove(k)

                    while len(standy_instances) >= pipeline_depth:
                        move_pipeline = random.sample(standy_instances, pipeline_depth)
                        for k in move_pipeline:
                            active_instances.add(k)
                            standy_instances.remove(k)

            curr_time += timedelta(seconds=5)
            f.write('========================================================================\n')
            f.write('\n\n')
    except KeyError as e:
        print(repr(e))

    print('done')
    return trace


exp_dir = get_next_dir()
os.makedirs(exp_dir)
exp_metadata = {
    'preempt_chance_per_minute': args.preemption_chance_per_minute,
    'preemption_distribution': args.preemption_distribution,
    'addition_chance_per_minute': args.addition_chance_per_minute,
    'addition_distribution': args.addition_distribution,
    'simulation_length_minutes': total_minutes,
}
exp_metadata_file = os.path.join(exp_dir, 'exp-info.json')
with open(exp_metadata_file, 'w') as f:
    json.dump(exp_metadata, f, indent=4)

trace = run_simulation(
    exp_metadata['preempt_chance_per_minute'], exp_metadata['preemption_distribution'],
    exp_metadata['addition_chance_per_minute'], exp_metadata['addition_distribution'],
    exp_metadata['simulation_length_minutes'],
    exp_dir)

trace_df = pd.DataFrame(trace)
trace_df.to_csv(os.path.join(exp_dir, 'simulation-trace.csv'))

#trace_file = os.path.join(exp_dir, 'simulation-trace.csv')
#trace_df = pd.read_csv(trace_file)
#
#in_service_file = os.path.join(exp_dir, 'in-service-trace.csv')
#in_service = pd.read_csv(in_service_file)


#### Using the trace to calculat about how much idle time there was
def filter_range(start, end, df, column='dt'):
    if start == end:
        end = end + timedelta(seconds=10)

    df = df[df[column] >= start]
    df = df[df[column] <= end]
    return df

def find_next(dt, df, column='dt'):
    tmp = df[df[column] >= dt].iloc[0]
    return tmp[column]

in_service_file = os.path.join(exp_dir, 'in-service-trace.csv')
in_service = pd.read_csv(in_service_file)
in_service['dt'] = in_service['timestamp'].apply(lambda x: ts_to_dt(x, "%Y-%m-%d %H:%M:%S"))

train_loss_time_tmp = trace_df[trace_df['event'] == 'step']
reconfig_times_tmp = trace_df[trace_df['event'] == 'reconfig']

tlt_dict = {
    'ws': train_loss_time_tmp['ws'].tolist(),
    'dt': train_loss_time_tmp['dt'].tolist(),
    'event': train_loss_time_tmp['event'].tolist(),
    'event-id': train_loss_time_tmp['event-id'].tolist()
}
train_loss_time = pd.DataFrame(tlt_dict)

reconfig_dict = {
    'ws': reconfig_times_tmp['ws'].tolist(),
    'dt': reconfig_times_tmp['dt'].tolist(),
    'event': reconfig_times_tmp['event'].tolist(),
    'event-id': reconfig_times_tmp['event-id'].tolist()
}
reconfig_times = pd.DataFrame(reconfig_dict)

prev_step = -1
prev_row = None
step = -1
wasted_work_ranges = []
for i in range(len(train_loss_time)):
    try:
        row = train_loss_time.loc[i]
        step = row['event-id']
        if prev_step >= step:
            ri = i
            tgt_step = step
            curr_step = prev_step
            end = prev_row['dt']
            while curr_step >= tgt_step:
                ri -= 1
                curr_row = train_loss_time.iloc[ri]
                curr_step = curr_row['event-id']

            start_row = train_loss_time.iloc[ri]
            start = start_row['dt']
            end = find_next(end, reconfig_times)
            wasted_work_ranges.append(filter_range(start, end, in_service))

        prev_row = row
        prev_step = step
    except Exception as e:
        print('ROW', row)
        raise e

reconfig_ranges = []
for i in range(len(trace_df)-1):
    row = trace_df.iloc[i]
    if row['event'] == 'reconfig':
        next_row = trace_df.iloc[i+1]
        start = row['dt']
        end = next_row['dt']
        reconfig_ranges.append(filter_range(start, end, in_service))

fig, ax = plt.subplots(figsize=(8, 6))
date_format = mdates.DateFormatter("%m-%d %H:%M")
ax.plot(in_service['dt'], in_service['count'])

ax.fill_between(in_service['dt'], in_service['count'])

## FILL IN HERE
for r in wasted_work_ranges:
    ax.fill_between(r['dt'], r['count'], color='orange')

for r in reconfig_ranges:
    ax.fill_between(r['dt'], r['count'], color='red')

ax.grid()
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
graph_fname = os.path.join(exp_dir, f'in-service.png')
plt.savefig(graph_fname, forma='png', dpi=240, bbox_inches='tight')


## This is a stupid way of approximating the time spent wasted.
## Took too long to coalesce ranges and its late so forget it
## This gets the idea across
wasted_datetimes = set()

for w in wasted_work_ranges:
    for ts, row in w.iterrows():
        wasted_datetimes.add(row['dt'])

for r in wasted_work_ranges:
    for ts, row in w.iterrows():
        wasted_datetimes.add(row['dt'])


wasted_time = len(wasted_datetimes) / len(in_service)

results_file = os.path.join(exp_dir, 'results.json')
results_dict = {
    'percent_time_wasted': wasted_time
}
json.dump(results_dict, open(results_file, 'w'), indent=4)
print(f'Results written to {exp_dir}')
