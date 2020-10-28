def aws_availability_command(options):
        from .aws_availability import run
        run()

def tutorial_mnist_command(options):
	from .tutorial_mnist import run
	run(options.worker_index)

def test_command(options):
        from .test import run
        run()
