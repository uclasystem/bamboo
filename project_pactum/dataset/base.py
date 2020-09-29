from project_pactum import BASE_DIR

def setup_datasets():
	from project_pactum.dataset.cifar import setup_cifar_10

	setup_cifar_10()
