import os, pathlib
import numpy as np
import pandas as pd
import tensorflow as tf

from models import ModelConstructor
from utils import load_dataset, saveDataFrame, calc_rf_indices

DEBUG = False

# constants
IMG_SIZE = (256, 256, 3)
NUM_CLASSES = 1000

NUM_AB = 24

# path settings
PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(PROJECT_PATH, "trained_models", "v1")
DATA_PATH = "/nobackup/users/ozaki/ImageNet/data/processed/test_tfrec"
BATCH_SIZE = 256
if DEBUG:
    RESULT_PATH = os.path.join(PROJECT_PATH, "result", "v0")
else:
    RESULT_PATH = os.path.join(PROJECT_PATH, "result", "v1")

def test_ablation(model_name="alexnet", list_models=[], 
                  ranking_indices=["color", "fft_freq", "fft_az"], 
                  train_version=0, repeat=0, overwrite=False):
    """
    This function tests the ablation of the given neural network model.

    Args:
        model_name (str): The name of the model to be used.
        list_models (list): A list of model files to be processed.
        ranking_indeces (list): A list of ranking indices to be used for ordering the filters.
        train_version (int): The version of the trained model to be used.
        repeat (int): The repetition index.
        overwrite (bool): If True, the existing results are overwritten.
    """
    
    # arg check
    assert isinstance(model_name, str), "model_name must be a string"
    assert isinstance(list_models, list), "list_models must be a list"
    assert isinstance(ranking_indices, list), "ranking_indices must be a list"
    assert isinstance(train_version, int), "train_version must be an integer"
    assert isinstance(repeat, int), "repeat must be an integer"
    assert isinstance(overwrite, bool), "overwrite must be a boolean"
    assert len(list_models) > 0, "list_models is empty"
    assert len(ranking_indices) > 0, "ranking_indices is empty"
    assert len(model_name) > 0, "model_name is empty"
    assert train_version >= 0, "train_version is not a positive integer"
    assert repeat >= 0, "repeat is not a positive integer"
    
    # path setup for trained models and results
    model_path = os.path.join(MODEL_PATH, model_name, str(train_version), str(repeat))
    result_path =  os.path.join(RESULT_PATH, model_name, str(train_version), str(repeat))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # testing conditions
    ablations = np.arange(0, NUM_AB + 1, 1)

    # model setup
    model = ModelConstructor(model_name=model_name, 
                              input_shape=IMG_SIZE, 
                              num_classes=NUM_CLASSES).getModel()

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc'])

    # setup dataframe
    acc_pd = pd.DataFrame(index=ablations)
    
    # load test dataset
    test_col = load_dataset(DATA_PATH, IMG_SIZE, color=True, blur_sigma=0)
    test_gra = load_dataset(DATA_PATH, IMG_SIZE, color=False, blur_sigma=0)
    test_col = test_col.batch(BATCH_SIZE).cache().prefetch(1)
    test_gra = test_gra.batch(BATCH_SIZE).cache().prefetch(1)

    # test ablation
    for model_file in list_models:
        print(model_file)
        tf.keras.backend.clear_session()
        model.load_weights(os.path.join(model_path, model_file))

        if "alexnet" in model_name:
            conv1_layer = model.layers[4]
        else:
            print("Not (yet) implemented for " + model_name)
            return
        
        # Calculate the ranking
        original_weights = conv1_layer.get_weights()[0]
        [color_index, fft_freq_index, fft_az_index] = calc_rf_indices(original_weights, return_rank=True)

        # test ablation
        for reverse_order in [True, False]:
            for ranking_index in ranking_indices:
                if ranking_index == "color":
                    index = color_index
                elif ranking_index == "fft_freq":
                    index = fft_freq_index
                elif ranking_index == "fft_az":
                    index = fft_az_index
                else:
                    raise ValueError("Invalid ranking_index: " + ranking_index)
                
                if reverse_order:
                    index = np.flip(index)
                    ranking_index = ranking_index + "_reverse"

                for ablation in ablations:
                    print("Currently working on ablation " + str(ablation))
                    modified_weights = np.array(original_weights)
                    modified_weights[:, :, :, index[:ablation]] = 0
                    conv1_layer.set_weights([modified_weights, conv1_layer.get_weights()[1]])
                    # test color
                    [loss, acc] = model.evaluate(x=test_col, verbose=0)
                    acc_pd.at[ablation, model_file + "_" + ranking_index + "_col"] = acc
                    # test grayscale
                    [loss, acc] = model.evaluate(x=test_gra, verbose=0)
                    acc_pd.at[ablation, model_file + "_" + ranking_index + "_gra"] = acc

    # save results
    saveDataFrame(acc_pd, os.path.join(result_path, "ablation_performance.csv"), overwrite=overwrite)


if __name__ == "__main__":
    test_ablation()