from test_texture_ablation import test_texture_ablation
from test_ablation import test_ablation
from calc_correlations import calc_correlations
from time import time
import os, glob, pathlib

PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(PROJECT_PATH, "trained_models", "v1")
overwrite = True

model_name = "alexnet22_48"
list_models = ["c0-100_c0-100", "g4-100_c0-100"]
train_version = 0

# Calculate correlations
comparison_pairs = [["c0", "g0"], ["c0", "c4"]]
for comparison_pair in comparison_pairs:
    for repeat in range(5):
        calc_correlations(model_name, list_models, train_version, repeat, comparison_pair)

# Test ablation
for repeat in range(5):
    test_ablation(model_name, list_models, ["color", "fft_freq"], train_version, repeat, overwrite)

# Test texture ablation
list_settings = [("alexnet22_48", 0, 48, 48), ("alexnet22_48", 1, 48, 48), ("alexnet22_48_half", 0, 48, 48), 
                 ("alexnet22", 0, 96, 48), ("alexnet", 0, 96, 12)]
for model, train_version, num_ab, n_top_col_pixel in list_settings:
    print(f"Model {model}")
    for repeat in range(5):
        print(f"Repeat {repeat}")
        model_path = os.path.join(MODEL_PATH, model, str(train_version), str(repeat))
        list_models = glob.glob(os.path.join(model_path, "*.index"))
        list_models = [os.path.basename(model_file).split(".")[0] for model_file in list_models]
        list_models.sort()

        start_time = time()
        test_texture_ablation(model, list_models, num_ab, n_top_col_pixel, ["color", "fft_freq"],
                              train_version, repeat, overwrite)
        end_time = time()
        print("Texture ablation test took " + str(end_time - start_time) + " seconds")
