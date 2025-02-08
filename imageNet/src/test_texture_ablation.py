import os, glob, pathlib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import re

from models import ModelConstructor
from utils import load_images, calc_rf_indices, saveDataFrame

DEBUG = False

# constants
IMG_SIZE = (256, 256, 3)
NUM_CLASSES = 1000

# path settings
PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(PROJECT_PATH, "trained_models", "v1")
DATA_PATH = "/nobackup/users/ozaki/ImageNet/data/texture_transfer_padded"
if DEBUG:
    RESULT_PATH = os.path.join(PROJECT_PATH, "result", "v0")
else:
    RESULT_PATH = os.path.join(PROJECT_PATH, "result", "v1")
CLASS_INDICES_PATH = os.path.join(PROJECT_PATH, "data", "categories16_class_indices.pkl")

def make_decision(probability, cate16_class_indices):
    max_value = -1
    category_decision = None
    for category in cate16_class_indices.keys():
        indices = cate16_class_indices[category]
        values = np.take(probability, indices)
        aggregated_value = np.max(values)
        if aggregated_value > max_value:
            max_value = aggregated_value
            category_decision = category
    return category_decision


def test_texture_ablation(model_name="alexnet", list_models=[], num_ab=48, n_top_col_pixel=48,
                          ranking_indeces=["color", "fft_freq"], train_version=0, repeat=0, overwrite=False):
    """
    This function tests the ablation of the given neural network model on texture transferred images.

    Args:
        model_name (str): The name of the model to be used.
        list_models (list): A list of model files to be processed.
        ranking_indeces (list): A list of ranking indices to be used for ablation.
        train_version (int): The param set of the trained model.
        repeat (int): The repetition index.
    """
    # arg check
    assert isinstance(model_name, str), "model_name must be a string"
    assert isinstance(list_models, list), "list_models must be a list"
    assert isinstance(num_ab, int), "num_ab must be an integer"
    assert isinstance(ranking_indeces, list), "ranking_indeces must be a list"
    assert isinstance(train_version, int), "train_version must be an integer"
    assert isinstance(repeat, int), "repeat must be an integer"
    assert len(list_models) > 0, "list_models is empty"
    assert len(ranking_indeces) > 0, "ranking_indeces is empty"
    assert len(model_name) > 0, "model_name is empty"
    assert train_version >= 0, "train_version is not a positive integer"
    assert repeat >= 0, "repeat is not a positive integer"
    assert num_ab >= 0, "num_ab is not a positive integer"
    
    # path setup for trained models
    model_path = os.path.join(MODEL_PATH, model_name, str(train_version), str(repeat))
    assert os.path.exists(model_path), "model_path does not exist"
    result_path = os.path.join(RESULT_PATH, model_name, str(train_version), str(repeat))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # load class indices
    with open(CLASS_INDICES_PATH, "rb") as f:
        cate16_class_indices = pickle.load(f)

    # load test dataset
    test_files = glob.glob(os.path.join(DATA_PATH, "*.png"))
    x_test, y_test = load_images(test_files, IMG_SIZE, color=True, texture=True)
    
    # model setup
    model = ModelConstructor(model_name=model_name, 
                              input_shape=IMG_SIZE, 
                              num_classes=NUM_CLASSES).getModel()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])

    # result pandas setup
    res_pd = pd.DataFrame(index=cate16_class_indices.keys())

    for model_file in list_models:
        tf.keras.backend.clear_session()
        print("Working on " + model_file)
        model.load_weights(os.path.join(model_path, model_file))

        if "alexnet" in model_name:
            conv1_layer = model.layers[4]
        else:
            print("Not (yet) implemented for " + model_name)
            return

        # Calculate the ranking
        original_weights = conv1_layer.get_weights()[0]
        [color_index, fft_freq_index, fft_az_index] = calc_rf_indices(original_weights, 
                                                                      n_top_col_pixel=n_top_col_pixel,
                                                                      return_rank=True)

        for reverse_order in [False, True]:
            for ranking_index in ranking_indeces:
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

                for ablation in range(num_ab + 1):
                    print("Currently working on ablation " + str(ablation))
                    modified_weights = np.array(original_weights)
                    modified_weights[:, :, :, index[:ablation]] = 0
                    conv1_layer.set_weights([modified_weights, conv1_layer.get_weights()[1]])

                    # make predictions
                    pred = model.predict(x=x_test, verbose=0)

                    # count shape-texture stats
                    count_dict = {}
                    for cate in cate16_class_indices.keys():
                        count_dict.update({cate: [0, 0, 0]})

                    for i in range(len(y_test)):
                        probabilities = pred[i]
                        assert len(probabilities) == NUM_CLASSES, "probabilities length is not correct"
                        cates = y_test[i].split("-")
                        shape_cate = re.sub('[^A-Za-z]+', '', cates[0])
                        text_cate = re.sub('[^A-Za-z]+', '', cates[1])
                        if shape_cate == text_cate:
                            continue

                        decision = make_decision(probabilities, cate16_class_indices)
                        count = count_dict[shape_cate]
                        if shape_cate == decision:
                            count[0] = count[0] + 1
                        elif text_cate == decision:
                            count[2] = count[2] + 1
                        else:
                            count[1] = count[1] + 1

                        count_dict.update({shape_cate: count})

                    res_pd[f"{ranking_index}_{model_file}_ablation_{ablation}"] = count_dict.values()

    save_name = os.path.join(result_path, "texture_ablation.csv")
    saveDataFrame(res_pd, save_name, overwrite=overwrite)

if __name__ == "__main__":
    test_texture_ablation()