import os, glob, pathlib, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from models import ModelConstructor
from utils import load_images, gaussianBlur

DEBUG = False

# Constantss
IMG_SIZE = (256, 256, 3)
NUM_CLASSES = 1000
BATCH_SIZE = 64

# Path settings
PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(PROJECT_PATH, "trained_models", "v1")
DATA_PATH = "/nobackup/users/ozaki/ImageNet/data/processed"
if DEBUG:
    RESULT_PATH = os.path.join(PROJECT_PATH, "result", "v0")
else:
    RESULT_PATH = os.path.join(PROJECT_PATH, "result", "v1")


def calc_correlations(model_name="alexnet", list_models=[], train_version=0, repeat=0, comparison_pair=["c0", "g4"],
                      list_layers=["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "dense", "dense_1", "dense_2"]):   
    """
    This function generates and saves activations for the given neural network models and its layers.

    Args:
        model_name (str): The name of the model to be used.
        list_models (list): A list of model files to be processed.
        train_version (int): The param set of the trained model to be used.
        repeat (int): The repetition index.
        comparison_pair (list): List of two conditions to compare. Format: ["c0", "g4"] where:
                                First char: 'c' for color, 'g' for grayscale, Number: blur sigma value
        list_layers (list): A list of layers to be processed.
    """

    # Arg check
    assert isinstance(model_name, str), "model_name must be a string"
    assert isinstance(list_models, list), "list_models must be a list"
    assert len(list_models) > 0, "list_models is empty"
    assert os.path.exists(MODEL_PATH), "MODEL_PATH does not exist"
    assert os.path.exists(DATA_PATH), "DATA_PATH does not exist"
    assert isinstance(train_version, int), "train_version must be an integer"
    assert isinstance(repeat, int), "repeat must be an integer"
    assert train_version >= 0, "train_version must be non-negative"
    assert repeat >= 0, "repeat must be non-negative"
    assert isinstance(list_layers, list), "list_layers must be a list"
    assert len(list_layers) > 0, "list_layers is empty"
    assert isinstance(comparison_pair, list), "comparison_pair must be a list"
    assert len(comparison_pair) == 2, "comparison_pair must contain two dictionaries"

    # Path setup for trained models and results
    model_path = os.path.join(MODEL_PATH, model_name, str(train_version), str(repeat))
    assert os.path.exists(model_path), "model_path does not exist"
    correlation_path =  os.path.join(RESULT_PATH, model_name, str(train_version), str(repeat), "correlations")
    if not os.path.exists(correlation_path):
        os.makedirs(correlation_path)

    # Load test dataset
    dataset_path =  os.path.join(DATA_PATH, "test3000")
    assert os.path.exists(dataset_path), "dataset_path does not exist"
    test_files = glob.glob(os.path.join(dataset_path, "*.JPEG"))
    assert len(test_files) == 3000, "The number of test files is incorrect"

    datasets = []
    for condition in comparison_pair:
        if condition[0] == "c":
            color = True
        elif condition[0] == "g":
            color = False
        else:
            raise ValueError("Invalid condition: " + condition)
        blur_sigma = int(condition[1:])
        x_test, y_test = load_images(test_files, IMG_SIZE, color=color)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(BATCH_SIZE)
        if blur_sigma > 0:
            test_dataset = test_dataset.map(lambda x, y: (gaussianBlur(x, blur_sigma), y))
        test_dataset = test_dataset.cache()
        datasets.append(test_dataset)

    # Model setup
    model = ModelConstructor(model_name=model_name, 
                             input_shape=IMG_SIZE, 
                             num_classes=NUM_CLASSES).getModel() 
    
    output_list = [model.get_layer(x).output for x in list_layers]
    aux_model = Model(inputs=model.inputs, outputs=output_list) 

    for model_file in list_models:
        tf.keras.backend.clear_session()
        print("Working on " + model_file)
        
        # Load weights
        model.load_weights(os.path.join(model_path, model_file))
        
        # Get activations
        activations = []
        for dataset in datasets:
            out = aux_model.predict(dataset, verbose=0)
            out = [np.array(x) for x in out]
            activations.append(out)

        # Compute correlations
        correlations_all = []
        for act1, act2 in zip(activations[0], activations[1]):
            assert act1.shape == act2.shape, "Activation shapes are different"
            act_shape = act1.shape
            num_units = act_shape[-1]
            length = int(np.prod(act_shape)/num_units)
            act1_flat = np.reshape(act1, (length, num_units))
            act2_flat = np.reshape(act2, (length, num_units))
            correlations = np.zeros(num_units)
            for i in range(num_units):
                corr = np.corrcoef(act1_flat[:, i], act2_flat[:, i])
                correlations[i] = corr[0, 1]
            correlations_all.append(correlations)
        
        # Save the correlations to a file
        save_name = f"corr_{model_file}_{comparison_pair[0]}_{comparison_pair[1]}.pkl"
        pickle.dump(correlations_all, open(os.path.join(correlation_path, save_name), "wb"))
        

if __name__ == "__main__":
    calc_correlations()