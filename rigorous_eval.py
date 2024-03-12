# import statements 
import sys
sys.path.insert(1, './BioAutoMATED/main_classes/')
sys.path.append('./BioAutoMATED')
from wrapper import run_bioautomated
from integrated_design_helpers import *
from generic_automl_classes import convert_generic_input, read_in_data_file
from generic_deepswarm import print_summary
from transfer_learning_helpers import transform_classification_target, transform_regression_target, fit_final_deepswarm_model
from generic_tpot import reformat_data_traintest
from sklearn.metrics import r2_score
import scipy.stats as sp
from keras.initializers import glorot_uniform
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
import autokeras
import pickle
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

def calculate_metrics(preds, y):
    """
    Calculate 'R2', 'Pearson', and 'Spearman' metrics.

    Parameters:
    - preds: Predicted values
    - y: True values

    Returns:
    - r2: R-squared (R2) score
    - pearson: Pearson correlation coefficient
    - spearman: Spearman rank correlation coefficient
    """
    # R-squared (R2) score
    r2 = r2_score(y, preds)

    # Pearson correlation coefficient
    pearson, _ = pearsonr(preds, y)

    # Spearman rank correlation coefficient
    spearman, _ = spearmanr(preds, y)

    return r2, pearson, spearman

output = {}
for col in ["Gene","Promoter","RBS"]:
    for i in range(1,6):
        result_list = []
        data_folder = './dataset/rigorous/'
        data_file = f'test_{col}_group{i}.csv'
        root_path = f"./ckpt/rigorous/{col}_group{i}"
        rawdata = pd.read_csv(data_folder+data_file)
        # rawdata.at[0, 'Seq'] = rawdata.iloc[0].Seq*5
        # rawdata.Seq = rawdata.Seq.apply(lambda x:x[:826])
        rawdata.to_csv('./output/experimental_data_fineturn.csv', index=False, encoding = 'utf_8_sig')

        # Load DeepSwarm Model and freeze all except last two layers (randomly chose this - feel free to customize)
        final_model_path = f'{root_path}/outputs/deepswarm/regression/'
        final_model_name = 'deepswarm_deploy_model.h5'
        # get sequences with help from https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
        with CustomObjectScope({'GlorotUniform': glorot_uniform(), 'BatchNormalizationV1': BatchNormalization()}): # , 'BatchNormalizationV1': BatchNormalization()
            model = tf.keras.models.load_model(final_model_path + final_model_name)
        print(model.summary())
        print('model is originally trainable: ' + str(model.trainable))
        print('number of layers in the model: ' + str(len(model.layers)))

        # set all layers except last two dense ones to be fixed
        for layer_idx, layer in enumerate(model.layers):
            if layer_idx > len(model.layers) - 3:
                print(str(layer_idx) + ': ' + str(layer) + ', keeping trainable = ' + str(layer.trainable))
            else:
                layer.trainable = False
                print(str(layer_idx) + ': ' + str(layer) + ', setting trainable to ' + str(layer.trainable))

        # Transform the test set RBS data to fine-tune this model
        data_folder = './output/'
        data_file = 'experimental_data_fineturn.csv'

        # Give inputs for data generation
        input_col = 'seq'
        target_col = 'target'
        pad_seqs = 'max'
        augment_data = 'none'
        sequence_type = 'nucleic_acid'
        task = 'regression'
        model_type = 'deepswarm'

        # allows user to interpret model with data not in the original training set
        # so apply typical cleaning pipeline
        df_data_input, df_data_output, _ = read_in_data_file(data_folder + data_file, input_col, target_col)
            
        # format data inputs appropriately for autoML platform    
        numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph = convert_generic_input(df_data_input, df_data_output, pad_seqs, augment_data, sequence_type, model_type = model_type)

        # transform output (target) into bins for classification
        transformed_output, transform_obj = transform_regression_target(df_data_output)
            
        # now, we have completed the pre-processing needed to feed our data into deepswarm
        # deepswarm input: numerical_data_input
        # deepswarm output: transformed_output
        X = numerical_data_input
        y = transformed_output

        # 使用微调前的模型进行预测
        preds = model.predict(X)
        r2, pearson, spearman = calculate_metrics(preds, y)
        result_list.append([r2,pearson[0],spearman])
        print(f"R2: {r2}")
        print(f"Pearson: {pearson}")
        print(f"Spearman: {spearman}")

        data_folder = './output/'
        data_file = 'experimental_data_fineturn.csv'


        # Give inputs for data generation
        input_col = 'seq'
        target_col = 'target'
        pad_seqs = 'max'
        augment_data = 'none'
        sequence_type = 'nucleic_acid'
        task = 'regression'
        model_type = 'autokeras'

        # allows user to interpret model with data not in the original training set
        # so apply typical cleaning pipeline
        df_data_input, df_data_output, _ = read_in_data_file(data_folder + data_file, input_col, target_col)
            
        # format data inputs appropriately for autoML platform    
        numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph = convert_generic_input(df_data_input, df_data_output, pad_seqs, augment_data, sequence_type, model_type = model_type)

        # Format data inputs appropriately for autoML platform
        transformed_output, transform_obj = transform_regression_target(df_data_output)

        # now, we have completed the pre-processing needed to feed our data into autokeras
        # autokeras input: oh_data_input
        # autokeras output: transformed_output
        X = oh_data_input
        y = transformed_output # don't convert to categorical for autokeras

        final_model_path = f'{root_path}/models/autokeras/regression/'
        final_model_name = 'optimized_autokeras_pipeline_regression.h5'

        clf = autokeras.utils.pickle_from_file(final_model_path+final_model_name)
        preds = clf.predict(np.array(X))
        r2 = r2_score(np.array(y), preds)
        print("R-squared after no retraining: ", r2)
        r2, pearson, spearman = calculate_metrics(preds, np.array(y).flatten())
        print(f"R2: {r2}")
        print(f"Pearson: {pearson}")
        print(f"Spearman: {spearman}")

        result_list.append([r2,pearson,spearman])


        # read in data file
        data_folder = './output/'
        data_file = 'experimental_data_fineturn.csv'

        # give inputs for data generation
        input_col_name = 'seq'
        target_col = 'target'
        pad_seqs = 'max'
        augment_data = 'none'
        sequence_type = 'nucleic_acid'
        task = 'regression'
        model_type = 'tpot'

        # allows user to interpret model with data not in the original training set
        # so apply typical cleaning pipeline
        df_data_input, df_data_output, _ = read_in_data_file(data_folder + data_file, input_col, target_col)
            
        # format data inputs appropriately for autoML platform    
        numerical_data_input, oh_data_input, df_data_output, scrambled_numerical_data_input, scrambled_oh_data_input, alph = convert_generic_input(df_data_input, df_data_output, pad_seqs, augment_data, sequence_type, model_type = model_type)

        # Format data inputs appropriately for autoML platform
        transformed_output, transform_obj = transform_regression_target(df_data_output)

        X = numerical_data_input
        y = transformed_output # don't convert to categorical for tpot
        X, y = reformat_data_traintest(X, y)

        # give inputs for paths
        final_model_path = f'{root_path}/outputs/tpot/regression/'
        final_model_name = 'final_model_tpot_regression.pkl'
        output_folder = final_model_path

        with open(final_model_path+final_model_name, 'rb') as file:  
            model = pickle.load(file)
            

        preds = model.predict(X)

        r2 = r2_score(np.array(y), preds)
        print('Original model on new test data R2 : ', r2)
        print('Original model on new test data: ', sp.pearsonr(y, preds))


        r2, pearson, spearman = calculate_metrics(preds, np.array(y).flatten())
        print(f"R2: {r2}")
        print(f"Pearson: {pearson}")
        print(f"Spearman: {spearman}")

        result_list.append([r2,pearson,spearman])
        output[root_path] = result_list
# 保存字典为.pkl文件
with open('output/rigorous.pkl', 'wb') as f:
    pickle.dump(output, f)