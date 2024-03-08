import sys
sys.path.insert(1, 'BioAutoMATED/main_classes/')
import warnings
warnings.filterwarnings("ignore")
from wrapper import run_bioautomated
import shutil
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# make directory where these results can live - no need to run if already have a directory

import tensorflow as tf
tf.test.is_gpu_available()

# specify parameters for the actual search (example)
max_runtime_minutes = 60 # time in minutes to give to each implemented AutoML algorithms
num_folds = 2 # recommend 3 - 5 folds for robustness

# Deepswarm execution
num_final_epochs = 10
yaml_params = {'ant_count': 1, 'max_depth': 2, 'epochs': 5}

# TPOT execution
num_generations = 5
population_size = 5


for col in ["Gene","Promoter","RBS"]:
    for i in range(1,6):
        data_folder = './dataset/rigorous/'
        data_file = f'train_{col}_group{i}.csv'
        input_col = 'seq'
        target_col = 'target'
        sequence_type = 'nucleic_acid'
        verbosity = 1
        task = 'regression' # binary_classification, multiclass_classification, regression

        # Specify target folders for saving models and results
        # Generic here - will add the tags specifying classification/regression
        # as well as specific for the AutoML tool being used (i.e. /tpot/)
        root_path = f"./ckpt/rigorous/{col}_group{i}"
        os.mkdir(root_path)
        model_folder = f'{root_path}/models/'
        output_folder = f'{root_path}/outputs/'

        run_bioautomated(task, data_folder, data_file, sequence_type, model_folder, output_folder, input_col=input_col, target_col=target_col, max_runtime_minutes=max_runtime_minutes, num_folds=num_folds, verbosity=verbosity, num_final_epochs=num_final_epochs, yaml_params=yaml_params, num_generations=num_generations, population_size=population_size)
