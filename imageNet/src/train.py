from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import os, pathlib, pickle
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
IMG_SIZE = (256, 256, 3)
NUM_CLASSES = 1000

SHUFFLE_BUFFER = 1000

# Define paths
PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent
if DEBUG:
    MODEL_PATH = os.path.join(PROJECT_PATH, "trained_models", "v0")
    DATA_PATH = os.path.join(PROJECT_PATH, "data", "small_dataset")
    VERBOSE = 1
    NUM_IMG = 5005*2
else:
    MODEL_PATH = os.path.join(PROJECT_PATH, "trained_models", "v1")
    DATA_PATH = "/nobackup/users/ozaki/ImageNet/data/processed"
    VERBOSE = 2
    NUM_IMG = 1281167

class CustomCallback(Callback):
    """
    A custom callback class that saves the optimizer configuration
    periodically during the model training.

    Attributes:
    save_name (str): The base name of the saved optimizer configuration files.
    save_freq (int): The frequency (in epochs) at which the optimizer configuration is saved.
    """

    def __init__(self, save_name, save_freq):
        """
        Initialize the CustomCallback class with the save_name and save_freq.

        Args:
        save_name (str): The base name of the saved optimizer configuration files.
        save_freq (int): The frequency (in epochs) at which the optimizer configuration is saved.
        """
        super(CustomCallback, self).__init__()
        self.save_freq = save_freq
        self.save_name = save_name

    def on_epoch_end(self, epoch, logs=None):
        """
        Method called at the end of each epoch.
        
        Args:
        epoch (int): The current epoch number.
        logs (dict): logs
        """
        # Check if the epoch number is a multiple of the save frequency
        if (epoch + 1) % self.save_freq == 0:
            # Save the optimizer configuration
            file_name = f"{self.save_name}{epoch + 1:03d}_opt_config.pkl"
            with open(file_name, 'wb') as handle:
                pickle.dump(self.model.optimizer.get_config(), handle, pickle.HIGHEST_PROTOCOL)


def train_model(model_name="alexnet", color=True, blur_sigma=0, start_file=None, load_opt_config=False,
                half_conv=[False, False], num_epochs=2, save_freq=1, train_version=0, repeat=0):
    """
    Function to train a specified model on the dataset, with various training configurations and
    callbacks. It includes support for training on distributed systems and handling large datasets.

    Args:
    model_name (str): Name of the model to train.
    color (bool): If True, the images are processed in color, otherwise in grayscale.
    blur_sigma (float): Standard deviation of Gaussian blur applied to the images.
    start_file (str): Filename of the saved weights to load before training.
    load_opt_config (bool): If True, load the optimizer configuration from a file.
    half_conv (list): list of two boolean values indicating whether the first and 
                second half of the first convolutional layer are trainable
                only used for alexnet22_48_half
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

    # Set up training parameters
    trainingParams = getTrainingParams(train_version)
    params = {'model_name' : model_name,
              'color': color,
              'blur_sigma': blur_sigma,
              'start_file': start_file if start_file else 'None',
              'load_opt_config': load_opt_config,
              'half_conv': str(half_conv),
              'num_epochs' : num_epochs,
              'model_path': model_path,
              'train_version': train_version,
              'repeat': repeat,
              'seed': seed,
              **trainingParams}
    
    # Set up Neptune.ai logging
    run = neptune.init_run(project='marin-oz/ParvoMagno')
    run["params"] = params
    if DEBUG:
        run["sys/tags"].add(['debug', model_name])
    else:
        run["sys/tags"].add(['satori', model_name])

    run["sys/tags"].add(['rev'])
    
   # Set up the optimizer
    if params['optimizer'] == "SGD":
        if start_file and load_opt_config:
            with open(os.path.join(model_path, start_file)+'_opt_config.pkl', 'rb') as handle:
                optimizer = SGD.from_config(pickle.load(handle))
                print("loaded optimizer config:" + str(optimizer.get_config()))
        else:
            optimizer = SGD(learning_rate=params['learning_rate'], momentum=params['momentum'], 
                            nesterov=params['nesterov'])
    elif params['optimizer'] == "Adam":
        if start_file and load_opt_config:
            with open(os.path.join(model_path, start_file)+'_opt_config.pkl', 'rb') as handle:
                optimizer = Adam.from_config(pickle.load(handle))
                print("loaded optimizer config:" + str(optimizer.get_config()))
        else:
            optimizer = Adam(learning_rate=params['learning_rate'])
    else:
        raise ValueError("optimizer should be one of [SDG, Adam]")

    # Set up the prefix for this training
    save_name = 'c' if color else 'g'
    save_name = save_name + str(blur_sigma) + '-'
    if start_file:
        save_name = start_file + '_' + save_name
    save_name = os.path.join(model_path, save_name)

    # Set up distributed training settings
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    global_batch_size = params['batch_size'] * strategy.num_replicas_in_sync

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE

    # Set the option experimental_deterministic = False to allow order-altering optimizations.
    options.experimental_deterministic = False

    # Calculate the number of steps per epoch
    print("Number of images:" +str(NUM_IMG))
    steps_per_epoch = int(np.ceil(NUM_IMG/global_batch_size))
    print("Steps per epoch:" +str(steps_per_epoch))

    # Load the train and val datasets
    train_dataset = load_dataset(os.path.join(DATA_PATH, "train_tfrec"), IMG_SIZE, color, blur_sigma)
    val_dataset = load_dataset(os.path.join(DATA_PATH, "test_tfrec"), IMG_SIZE, color, blur_sigma)
    
    train_dataset = train_dataset.with_options(options).cache()\
                                 .shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)\
                                 .batch(global_batch_size).prefetch(AUTO)    
    val_dataset = val_dataset.with_options(options).batch(global_batch_size).cache().prefetch(AUTO)

    # Construct a model to train
    with strategy.scope():
        modelConstructor = ModelConstructor(model_name=model_name, 
                                            input_shape=IMG_SIZE, 
                                            num_classes=NUM_CLASSES,
                                            half_conv_train=half_conv)
        model = modelConstructor.getModel()
        modelConstructor.printSummary()
        model.compile(optimizer=optimizer,
                      loss=SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        if start_file:
            model.load_weights(os.path.join(model_path, start_file)).expect_partial()
    
    # Set up callback functions
    checkpoint_filepath = save_name + '{epoch:03d}'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, 
                                                save_best_only=False, save_freq=int(steps_per_epoch*save_freq))
    neptune_cbk = NeptuneCallback(run=run)
    list_callbacks = [model_checkpoint_callback, neptune_cbk]

    if params['reduce_lr']:
        reduce_lr_cbk = ReduceLROnPlateau(monitor=params['rlr_monitor'], factor=params['rlr_factor'], 
                                          patience=params['rlr_patience'], verbose=1, 
                                          mode=params['rlr_mode'], min_delta=params['rlr_min_delta'], 
                                          cooldown=params['rlr_cooldown'], min_lr=params['rlr_min_lr'])
        list_callbacks.append(reduce_lr_cbk)
        custum_cbk = CustomCallback(save_name=save_name, save_freq=save_freq)
        list_callbacks.append(custum_cbk)

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
