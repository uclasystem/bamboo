import project_pactum
from project_pactum.aws.instance import create_instance
from project_pactum.experiment.instance import Instance

import boto3
import time
import subprocess
import sys
import os

NFSID='i-01ed34ecd79d8d980'


def start_nfs_server():
    print("Starting NFS server")
    ec2 = boto3.client('ec2')
    nfs_server = Instance(NFSID)
    nfs_server.start()
    nfs_server.init_from_id()
    nfs_server.wait_for_ssh()
    return nfs_server

def get_horovod_options(options, instances):
    np = options.cluster_size * options.ngpus
    cluster_conf = ','.join([
        ':'.join([inst.private_ip, str(options.ngpus)])
        for inst in instances])

    return np, cluster_conf


def run(options):
    nfs_server = start_nfs_server()

    print("Allocating spot instances for workers")
    instances = create_instance(options.cluster_size, options.instance_type,
                                options.az, 'ami-0f0aa914da49f348e')

    try:
        time.sleep(5)
        instances = [Instance(i.id) for i in instances]
        for inst in instances:
            inst.init_from_id()

        print("Waiting for SSH connections to all instances")
        for inst in instances:
            inst.wait_for_ssh()

        print("Checking to make sure every server can access the NFS drive")
        for inst in instances:
            if 'No NFS mount' in inst.ssh_command('nfsiostat').stdout.decode('utf-8'):
                raise Exception("No NFS volume found on instance {}. "
                                "Make sure NFS server is running".format(inst.id))

        leader = instances[0]

        # Select the first instance to issue the run cmd
        print("Running imagenet")
        ret = leader.ssh_command(' '.join(['cd project-pactum; . .venv/bin/activate;',
            'git pull',
            'python -m project_pactum --daemonize',
            'experiment imagenet-pretrain --worker',
            '--ngpus', str(options.ngpus),
            '--cluster-size', str(options.cluster_size),
            '--epochs', str(options.epochs),
            '--workers', get_horovod_options(options, instances)[1]]))

        print("STDERR", ret.stderr.decode('utf-8'))
        print("STDOUT", ret.stdout.decode('utf-8'))

        if (ret.returncode != 0):
            raise Exception

    except Exception as e:
        print("[ERROR]", str(e))
        print("Terminating instances")
        ec2 = boto3.client('ec2')
        ec2.terminate_instances(InstanceIds=[i.id for i in instances])

        #nfs_server.stop()


## All this should happen on the remote server inside the '--worker' operations
def create_log_folder():
    log_dir = os.path.join('/home', 'project-pactum', 'experiment', 'imagenet-pretrain')

    subprocess.run(('mkdir -p ' + log_dir).split())
    return log_dir

def construct_run_cmd(options, log_dir):
    np = options.ngpus * options.cluster_size
    horovod_run_cmd = ' '.join(['cd /home/project-pactum/project-pactum;',
        'horovodrun', '-np', str(np), '-H', options.workers,
        'python project_pactum/experiment/pytorch_imagenet_resnet50.py',
        '--epochs', str(options.epochs)])

    return horovod_run_cmd

def worker(options):
    log_dir = create_log_folder()
    horovod_run_cmd = construct_run_cmd(options, log_dir)
    with open(log_dir + '/output.txt', 'w') as f:
        subprocess.run(horovod_run_cmd, stdout=f, stderr=f, shell=True)
