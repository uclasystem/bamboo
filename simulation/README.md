# Simulation Framework

There are two main files here:
+ Checkpoint-Simulation-Framework.ipynb
+ checkpoint-simulation-framework.py

The `ipynb` file is a Python notebook containing a set of experiments I ran
to come up with the way to generate preepmtion traces.

The `py` file contains an automated way of running through the genreation of
a simulation of asynchronous checkpointing.
It uses the performance profiling I did on a 4B parameter model with 16 nodes
and simulates the behavior of the checkpoint manager over time as nodes join
and leave.
It outputs the wasted time of the simulated trace based on the number of
preemptions.

## Simultaion Framework Parameters
The simulation framework currently runs using the following parameters:
+ preemption chance per minute (float): chance of getting a set of preemptions
	every minute
+ preemption distribution (dict): A python dictionary of the format
	{ "mean": x, "std": y } where (x, y) are the mean and standard deviation
	of a normal distribution that the number of preemptions should be
	sampled from
+ addition chance per minute (float): Same as preemption chance but for adding
	nodes
+ addition distribution (dict): Same as preemption distribution but for additions
+ total minutes (int): Number of minutes to run the simulation

## Example
python checkpoint-simulation-framework.py -pcm .05 -pd "{ \"mean\": 0, \"std\": 1 }" -acm .10 -ad "{ \"mean\": 0, \"std\": 1}"

Results will be output to `simulation_results/run-{run #}`
