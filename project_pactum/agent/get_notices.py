import os
import requests
import signal
import time

def get_auth_token(ttl):
    get_token_header = {
        "X-aws-ec2-metadata-token-ttl-seconds": str(ttl)
    }
    response = requests.put("http://169.254.169.254/latest/api/token", headers=get_token_header)

    token = ''
    if response.status_code == 200:
        token = response.content
    else:
        print('Getting token failed with code {}'.format(response.status_code))

    return token

def check_for_preemption():
    token = get_auth_token(21600)

    while True:
        get_action_header = {
            "X-aws-ec2-metadata-token": token,
        }
        response = requests.get('http://169.254.169.254/latest/meta-data/spot/instance-action', headers=get_action_header)

        http_code = response.status_code

        if http_code == 401:
            token = get_auth_token(30)
        elif http_code == 200:
            os.kill(os.getpid(), signal.SIGTERM)
            break
        else:
            pass

        time.sleep(3)