import signal, os, time, urllib.request

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
        with urllib.request.urlopen('https://laforge.cs.ucla.edu') as f:
            print(f.read(100).decode('utf-8'), flush=True)
        break
    else:
        print('No warning', flush=True)
    time.sleep(1)
