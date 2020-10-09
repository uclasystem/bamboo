def tutorial_mnist_command(options):
	from project_pactum.experiment.tutorial_mnist import run, run_host
	if options.host:
		run_host()
	else:
		run()
