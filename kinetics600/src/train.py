from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import os, pathlib
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
import numpy as np
import tensorflow as tf
import time

from models import ModelConstructor
from params import getTrainingParams
from utils import load_dataset

AUTO = tf.data.AUTOTUNE

DEBUG = False

# Define constants
VIDEO_SIZE = (48, 128, 128, 3)
NUM_CLASSES = 600

SHUFFLE_BUFFER = 1000

# Define paths
PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent
if DEBUG:
    MODEL_PATH = os.path.join(PROJECT_PATH, "trained_models", "v0")
    DATA_PATH = os.path.join(PROJECT_PATH, "data", "small")
    CACHE_PATH = os.path.join(PROJECT_PATH, "data", "small", "cache")
    VERBOSE = 1
    NUM_VIDEOS = 256
else:
    MODEL_PATH = os.path.join(PROJECT_PATH, "trained_models", "v1")
    DATA_PATH = "/nobackup/users/ozaki/kinetics600_s/data"
    CACHE_PATH = os.path.join(PROJECT_PATH, "data", "cache")
    VERBOSE = 2
    NUM_VIDEOS = 424885

def train_model(model_name="alexnet_3d", color=True, blur_sigma=0, start_file=None,
                num_epochs=2, save_freq=1, train_version=0, repeat=0):
    """
    Function to train a specified model on a video dataset, with various training configurations and
    callbacks. It includes support for training on distributed systems and handling large datasets.

    Args:
    model_name (str): Name of the model to train.
    color (bool): If True, the videos are processed in color, otherwise in grayscale.
    blur_sigma (float): Standard deviation of Gaussian blur applied to the videos.
    start_file (str): Filename of the saved weights to load before training.
    num_epochs (int): Number of epochs for training.
    save_freq (int): Frequency (in epochs) at which the model weights and optimizer configuration are saved.
    train_version (int): Version number of the training configuration.
    repeat (int): repetition index
    """
    # Set up seed
    seed  = int(time.time()) # Based on current time
    tf.random.set_seed(seed)
    
    # Set up paths for trained models
    model_path = os.path.join(MODEL_PATH, model_name, str(train_version), str(repeat))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Set up cache path
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    # Set up training parameters
    trainingParams = getTrainingParams(train_version)
    params = {'model_name' : model_name,
              'color': color,
              'blur_sigma': blur_sigma,
              'start_file': start_file if start_file else 'None',
              'num_epochs' : num_epochs,
              'model_path': model_path,
              'train_version': train_version,
              'repeat': repeat,
              'seed': seed,
              **trainingParams}
    
    # Set up Neptune.ai logging
    run = neptune.init_run(project='marin-oz/3DCNN')
    run["params"] = params
    if DEBUG:
        run["sys/tags"].add(['debug', model_name])
    else:
        run["sys/tags"].add(['satori', model_name])
    run["sys/tags"].add(['rev'])
    
   # Set up the optimizer
    if params['optimizer'] == "SGD":
        optimizer = SGD(learning_rate=params['learning_rate'], momentum=params['momentum'], 
                        nesterov=params['nesterov'])
    elif params['optimizer'] == "Adam":
        optimizer = Adam(learning_rate=params['learning_rate'])
    else:
        raise ValueError("optimizer should be one of [SDG, Adam]")

    # Set up the prefix for this training
    save_name = 'c' if color else 'g'
    save_name = save_name + str(blur_sigma) + '-'
    if start_file:
        save_name = start_file + '_' + save_name
    save_name = os.path.join(model_path, save_name)

    # Set up data and cache paths
    prefix = 'col_' if color else 'gra_'
    prefix = prefix + 'blur' + str(blur_sigma)
    cache_path = os.path.join(CACHE_PATH, prefix)
    if prefix == 'col_blur0':
        data_path = os.path.join(DATA_PATH, "tfrecords")
    else:
        data_path = os.path.join(DATA_PATH, "tfrecords_modified", prefix)
        if not os.path.exists(data_path):
            raise ValueError("Modified tfrecords are not found")

    # Set up distributed training settings
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    global_batch_size = params['batch_size'] * strategy.num_replicas_in_sync

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # Set the option experimental_deterministic = False to allow order-altering optimizations.
    options.experimental_deterministic = False

    # Calculate the number of steps per epoch
    print("Number of videos:" +str(NUM_VIDEOS))
    steps_per_epoch = int(np.ceil(NUM_VIDEOS/global_batch_size))
    print("Steps per epoch:" +str(steps_per_epoch))

    # Load the train and val datasets
    train_dataset = load_dataset(data_path, train=True, shape=(VIDEO_SIZE[1], VIDEO_SIZE[2], VIDEO_SIZE[3]))
    val_dataset = load_dataset(data_path, train=False, shape=(VIDEO_SIZE[1], VIDEO_SIZE[2], VIDEO_SIZE[3]))

    train_dataset = train_dataset.with_options(options).cache()\
                                 .shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)\
                                 .batch(global_batch_size).prefetch(AUTO)    
    val_dataset = val_dataset.with_options(options).cache(cache_path).batch(global_batch_size).prefetch(AUTO)

    # Construct a model to train
    with strategy.scope():
        modelConstructor = ModelConstructor(model_name=model_name, 
                                            input_shape=VIDEO_SIZE, 
                                            num_classes=NUM_CLASSES)
        model = modelConstructor.getModel()
        modelConstructor.printSummary()
        model.compile(optimizer=optimizer,
                      loss=SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        if start_file:
            model.load_weights(os.path.join(model_path, start_file)).expect_partial()
    
    # Set up callback functions
    checkpoint_filepath = save_name + '{epoch:02d}'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, 
                                                save_best_only=False, save_freq=int(steps_per_epoch*save_freq))
    neptune_cbk = NeptuneCallback(run=run)
    list_callbacks = [model_checkpoint_callback, neptune_cbk]

    # Train the model
    model.fit(x=train_dataset,
              epochs=num_epochs,
              verbose=VERBOSE,
              validation_data=val_dataset,
              callbacks=list_callbacks)
    
    # Stop the Neptune.ai logging
    run.stop()


if __name__ == '__main__':
    train_model()
