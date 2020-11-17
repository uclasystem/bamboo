def aws_availability_command(options):
	from .aws_availability import analyze_daily, monitor
	if not options.skip_monitor:
		monitor()
	if options.analyze_daily:
		analyze_daily()

def tutorial_mnist_command(options):
	from .tutorial_mnist import run
	run(options.worker_index)

def test_command(options):
	from .test import run
	run()

def imagenet_pretrain_command(options):
    from .imagenet_pretrain import run
    if not options.worker:
        run(options)
    else:
        worker(options)
