import project_pactum
from project_pactum.aws.instance import create_instance

import boto3
import time
import subprocess
import sys

NFSID='i-0c4e34d951518f690'

def get_inst_info(aws_response):
    return aws_response['Reservations'][0]['Instances'][0]

class Instance:
    """
    Class for an AWS EC2 node
    """
    def __init__(self, inst_id, inst_type=None, placement=None,
                 private_ip=None, public_ip=None,
                 username='project-pactum', ssh_key='~/.ssh/pactum-key_rsa'):
        self.id = inst_id
        self.type = inst_type
        self.placement = placement
        self.private_ip = private_ip
        self.public_ip = public_ip
        self.user = username
        self.key = ssh_key

    def init_from_id(self):
        """
        If class only has ID, initialize the rest of the data for the instance
        """
        ec2 = boto3.client('ec2')
        response = ec2.describe_instances(InstanceIds=[self.id])
        instance_info = response['Reservations'][0]['Instances'][0]
        self.type = instance_info['InstanceType']
        self.placement = instance_info['Placement']['AvailabilityZone']
        self.private_ip = instance_info['PrivateIpAddress']
        if instance_info['State']['Name'] == 'running':
            self.public_ip = instance_info['PublicIpAddress']

    def start(self):
        ec2 = boto3.client('ec2')
        response = ec2.start_instances(InstanceIds=[self.id])
        print("Previous state:",
              response['StartingInstances'][0]['PreviousState']['Name'])
        print("Current state: ",
              response['StartingInstances'][0]['CurrentState']['Name'])

    def stop(self):
        self.public_ip = None
        ec2 = boto3.client('ec2')
        response = ec2.stop_instances(InstanceIds=[self.id])
        print("Previous state:",
              response['StoppingInstances'][0]['PreviousState']['Name'])
        print("Current state: ",
              response['StoppingInstances'][0]['CurrentState']['Name'])

    def reboot(self):
        print("Rebooting instance " + self.inst_id + "...")
        ec2 = boto3.client('ec2')
        ec2.reboot_instances(InstanceIds=[self.id])

    def get_public_ip(self):
        ec2 = boto3.client('ec2')
        resp = ec2.describe_instances(InstanceIds=[self.id])
        inst_state = get_inst_info(resp)['State']['Name']
        if inst_state.lower() in ['terminated', 'stopped', 'shutting-down']:
            raise Exception("Public IP not available in state" + inst_state)
        elif inst_state.lower() == 'running':
            return get_inst_info(resp)['PublicIpAddress']

        while inst_state.lower() == 'pending':
            time.sleep(5)
            resp = ec2.describe_instances(InstanceIds=[self.id])
            inst_state = get_inst_info(resp)['State']['Name']

        return get_inst_info(resp)['PublicIpAddress']

    def wait_for_ssh(self):
        if self.public_ip == None:
            self.public_ip = self.get_public_ip()

        ssh_command = ['ssh', '-q', '-t', '-i', self.key,
                       ''.join([self.user, '@', self.public_ip]), 'exit']
        retcode = subprocess.run(ssh_command).returncode

        start = time.time()
        while retcode != 0:
            time.sleep(5)
            retcode = subprocess.run(ssh_command).returncode

            if time.time() - start > 180:
                raise Exception("SSH timed out after 3 minutes")

        return 0

    def ssh_command(self, command, live=False):
        ssh_command = ['ssh', '-q', '-t', '-i', self.key,
                       ''.join([self.user, '@', self.public_ip])] + command.split()

        args = { 'stderr': sys.stderr, 'stdout': sys.stdout } if live\
               else { 'capture_output': True }
        return subprocess.run(ssh_command, **args)


def start_nfs_server():
    print("Starting NFS server")
    ec2 = boto3.client('ec2')
    nfs_server = Instance(NFSID)
    nfs_server.start()
    nfs_server.init_from_id()
    nfs_server.wait_for_ssh()

def get_horovod_options(options, instances):
    np = options.cluster_size * options.ngpus
    cluster_conf = ','.join([
        ':'.join([inst.private_ip, str(options.ngpus)])
        for inst in instances])

    return np, cluster_conf

def construct_run_cmd(options, instances):
    np, horovod_cluster_str = get_horovod_options(options, instances)
    horovod_run_cmd = ' '.join(['cd horovod-examples/pytorch;',
        '. .venv/bin/activate;', 'horovodrun -np', str(np), '-H',
        horovod_cluster_str, 'python pytorch_imagenet_resnet50.py --epochs',
        str(options.epochs)])

    return horovod_run_cmd

def run(options):
    start_nfs_server()

    print("Allocating spot instances for workers")
    instances = create_instance(options.cluster_size, options.instance_type,
                                options.az, 'ami-0bbb8d8530da66d8b')

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

        horovod_run_cmd = construct_run_cmd(options, instances)

        # Select the first instance to issue the run cmd
        print("Running imagenet")
        leader = instances[0]
        leader.ssh_command(horovod_run_cmd, True)

    except Exception as e:
        print("[ERROR]", str(e))
    finally:
        print("Terminating instances")
        ec2 = boto3.client('ec2')
        ec2.terminate_instances(InstanceIds=[i.id for i in instances])

        #nfs_server.stop()
