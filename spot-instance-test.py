import signal, os, time

warned = False

def handler(signum, frame):
    global warned
    print('Got signal:', signum, flush=True)
    warned = True

signal.signal(signal.SIGTERM, handler)

print('Starting with PID:', os.getpid(), flush=True)
while True:
    if warned:
        print('Found warning', flush=True)
    else:
        print('No warning', flush=True)
    time.sleep(1)
