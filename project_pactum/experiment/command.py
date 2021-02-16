HELP = None

def add_arguments(parser):
	subparsers = parser.add_subparsers(metavar='command')

	from project_pactum.experiment.command import aws_availability_command
	aws_availability_parser = subparsers.add_parser('aws-availability', help=None)
	aws_availability_parser.set_defaults(command=aws_availability_command)
	aws_availability_parser.add_argument('--skip-monitor', action='store_true')
	aws_availability_parser.add_argument('--analyze-daily', action='store_true')

	from project_pactum.experiment.command import test_command
	test_parser = subparsers.add_parser('test', help=None)
	test_parser.set_defaults(command=test_command)

	from project_pactum.experiment.command import tutorial_mnist_command
	tutorial_mnist_parser = subparsers.add_parser('tutorial-mnist', help=None)
	tutorial_mnist_parser.set_defaults(command=tutorial_mnist_command)
	tutorial_mnist_parser.add_argument('--worker-index', type=int, default=0)

	from project_pactum.experiment.command import imagenet_pretrain_command
	imagenet_parser = subparsers.add_parser('imagenet-pretrain', help=None)
	imagenet_parser.set_defaults(command=imagenet_pretrain_command)
	imagenet_parser.add_argument('--cluster-size', type=int, default=1)
	imagenet_parser.add_argument('--instance-type', type=str, default='p2.xlarge')
	imagenet_parser.add_argument('--ngpus', type=int, default=1)
	imagenet_parser.add_argument('--az', type=str, default=None)
	imagenet_parser.add_argument('--epochs', type=int, default=1)

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
    from .imagenet_pretrain import run, worker, status
    if options.worker:
        worker(options)
    else:
        run(options)
