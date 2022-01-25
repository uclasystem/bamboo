import signal, os, time

warned = False

def handler(signum, frame):
    global warned
    print('Got signal:', signum)
    warned = True

signal.signal(signal.SIGTERM, handler)

print('Starting with PID:', os.getpid())
while True:
    if warned:
        print('Found warning')
    else:
        print('No warning')
    time.sleep(1)