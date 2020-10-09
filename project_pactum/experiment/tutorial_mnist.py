import tensorflow_datasets as tfds
import tensorflow as tf

import json
import os
import pathlib
import subprocess

import project_pactum

# Copied from https://www.tensorflow.org/tutorials/distribute/keras

def scale(image, label):
	image = tf.cast(image, tf.float32)
	image /= 255

	return image, label

def decay(epoch):
	if epoch < 3:
		return 1e-3
	elif epoch >= 3 and epoch < 7:
		return 1e-4
	else:
		return 1e-5

def run_host():
	home_dir = str(pathlib.Path.home())
	key_file = os.path.join(home_dir, '.ssh', 'project-pactum')

	public_ips = project_pactum.aws.instance.get_public_ips()
	private_ips = project_pactum.aws.instance.get_private_ips()

	tf_configs = {}
	for i, public_ip in enumerate(public_ips):
                tf_configs[public_ip] = json.dumps({
			"cluster": {
				"worker": ['{}:55432'.format(x) for x in private_ips],
			},
			"task": {"type": "worker", "index": i}
		})

	ps = []
	for public_ip in public_ips:
		ssh_cmd = ' '.join(['ssh', '-i', key_file, 'project-pactum@{}'.format(public_ip), ''])
		subprocess.run(ssh_cmd + 'cd project-pactum && git pull --ff-only', shell=True)
		print(ssh_cmd + '"cd project-pactum && git pull --ff-only"')
		p = subprocess.Popen(ssh_cmd + '"cd project-pactum && TF_CONF=\'{}\' ~/venv/bin/python3 -m project_pactum experiment tutorial-mnist"'.format(tf_configs[public_ip]), shell=True)
		ps.append(p)
	for p in ps:
		p.wait()

def run():
	experiment_dir = os.path.join(project_pactum.BASE_DIR, 'experiment', 'tutorial-mnist')

	datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
	mnist_train, mnist_test = datasets['train'], datasets['test']

	strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
	print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

	num_train_examples = info.splits['train'].num_examples
	num_test_examples = info.splits['test'].num_examples

	BUFFER_SIZE = 10000

	BATCH_SIZE_PER_REPLICA = 64
	BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

	train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

	with strategy.scope():
		model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
			tf.keras.layers.MaxPooling2D(),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(64, activation='relu'),
			tf.keras.layers.Dense(10)
		])

	model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              optimizer=tf.keras.optimizers.Adam(),
	              metrics=['accuracy'])

                
	checkpoint_dir = os.path.join(experiment_dir, 'training_checkpoints')
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

	class PrintLR(tf.keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs=None):
			print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
		                                                  model.optimizer.lr.numpy()))

	callbacks = [
		tf.keras.callbacks.TensorBoard(log_dir=os.path.join(experiment_dir, 'logs')),
		tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
		                                   save_weights_only=True),
		tf.keras.callbacks.LearningRateScheduler(decay),
		PrintLR()
	]

	model.fit(train_dataset, epochs=12, callbacks=callbacks)

	model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
	eval_loss, eval_acc = model.evaluate(eval_dataset)
	print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

	path = os.path.join(experiment_dir, 'saved_model')
	model.save(path, save_format='tf')
