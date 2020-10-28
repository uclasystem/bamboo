import json
import os
import pathlib
import subprocess

import project_pactum

import tensorflow as tf

# Copied from https://www.tensorflow.org/tutorials/distribute/keras
# Copied from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras

def _is_chief(task_type, task_id):
	return task_type is None or task_type == 'chief' or (task_type == 'worker' and task_id == 0)

def load_data(batch_size):
	import numpy as np
	(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
	x_train = x_train / np.float32(255)
	y_train = y_train.astype(np.int64)
	train_dataset = tf.data.Dataset.from_tensor_slices(
		(x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
	return train_dataset

def _get_temp_dir(dirpath, task_id):
	base_dirpath = 'workertemp_' + str(task_id)
	temp_dir = os.path.join(dirpath, base_dirpath)
	tf.io.gfile.makedirs(temp_dir)
	return temp_dir

def write_dir(d, task_type, task_id):
	dirname = os.path.dirname(d)
	basename = os.path.basename(d)
	if not _is_chief(task_type, task_id):
		dirname = _get_temp_dir(d, task_id)
	return os.path.join(dirname, basename)

def build_and_compile_model():
	model = tf.keras.Sequential([
		tf.keras.Input(shape=(28, 28)),
		tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
		tf.keras.layers.Conv2D(32, 3, activation='relu'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(10)
	])
	model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              optimizer=tf.keras.optimizers.Adam(),
	              metrics=['accuracy'])
	return model

def run(worker_index):
	project_pactum.core.base.setup_tensorflow()

	experiment_dir = os.path.join(project_pactum.BASE_DIR, 'experiment', 'tutorial-mnist')

	tf_config = {
		'cluster': {
			'worker': ['localhost:12345', 'localhost:23456']
		},
		'task': {'type': 'worker', 'index': worker_index}
	}
	os.environ['TF_CONFIG'] = json.dumps(tf_config)
	strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
		communication=tf.distribute.experimental.CollectiveCommunication.AUTO
	)

	BATCH_SIZE_PER_REPLICA = 64
	BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
	dataset = load_data(BATCH_SIZE)

	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF 
	dataset_no_auto_shard = dataset.with_options(options)

	task_type = strategy.cluster_resolver.task_type if strategy.cluster_resolver else None
	task_id = strategy.cluster_resolver.task_id if strategy.cluster_resolver else None

        # Tensorboard logging
	log_dir = os.path.join(experiment_dir, 'log')
	write_log_dir = write_dir(log_dir, task_type, task_id)
	if os.path.exists(write_log_dir):
		tf.io.gfile.rmtree(write_log_dir)
	callbacks = [
		tf.keras.callbacks.TensorBoard(log_dir=write_log_dir,
		                               histogram_freq=0),
	]

	with strategy.scope():
		model = build_and_compile_model()

        # Checkpointing
	checkpoint_dir = os.path.join(experiment_dir, 'ckpt')
	checkpoint = tf.train.Checkpoint(model=model)
	write_checkpoint_dir = write_dir(checkpoint_dir, task_type, task_id)
	checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=write_checkpoint_dir, max_to_keep=1)
	def checkpoint_save():
		checkpoint_manager.save()
		if not _is_chief(task_type, task_id):
			tf.io.gfile.rmtree(write_checkpoint_dir)
	def checkpoint_restore():
		latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
		checkpoint.restore(latest_checkpoint)

	# Training
	with strategy.scope():
		checkpoint_restore()
	training = model.fit(dataset_no_auto_shard, epochs=5, steps_per_epoch=200, callbacks=callbacks, verbose=0)
	checkpoint_save()
