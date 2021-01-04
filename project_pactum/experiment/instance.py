import project_pactum

import boto3
import time
import subprocess
import sys

class Instance:
    """
    Class for an AWS EC2 node
    """
    def __init__(self, inst_id, inst_type=None, placement=None,
                 private_ip=None, public_ip=None,
                 username='project-pactum', ssh_key='~/.ssh/project-pactum'):
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
        inst_state = self.get_inst_info(resp)['State']['Name']
        if inst_state.lower() in ['terminated', 'stopped', 'shutting-down']:
            raise Exception("Public IP not available in state" + inst_state)
        elif inst_state.lower() == 'running':
            return self.get_inst_info(resp)['PublicIpAddress']

        while inst_state.lower() == 'pending':
            time.sleep(5)
            resp = ec2.describe_instances(InstanceIds=[self.id])
            inst_state = self.get_inst_info(resp)['State']['Name']

        return self.get_inst_info(resp)['PublicIpAddress']

    def wait_for_ssh(self):
        if self.public_ip == None:
            self.public_ip = self.get_public_ip()

        ssh_command = ['ssh', '-q', '-i', self.key,
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
        ssh_command = ['ssh', '-i', self.key,
                       ''.join([self.user, '@', self.public_ip]),
                       '"bash -c \'', command, '\'"']
        ssh_command = ' '.join(ssh_command)

        args = { 'stderr': sys.stderr, 'stdout': sys.stdout } if live\
               else { 'capture_output': True }
        print("SSH Command:", ssh_command)
        return subprocess.run(ssh_command, shell=True, **args)

    def get_inst_info(self, aws_response):
        return aws_response['Reservations'][0]['Instances'][0]
