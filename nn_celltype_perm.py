import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import tensorflow as tf
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor

def main():
    # Specify the path to your VCF file
    vcf_file = "/home/asr94/project/data/genotype_data/L2.3.IT/L2.3.IT_3.vcf"

    # # Read the VCF file into a pandas DataFrame
    vcf_data = pd.read_csv(vcf_file, sep='\t', skiprows=1)
    vcf_columns = vcf_data.columns
    vcf_columns = vcf_columns[9:] # just the sample ids in a list

    vcf_array = vcf_data.values # converting to numpy array

    # Specify the path to your BED file
    bed_file = "/home/asr94/project/data/expr_data/L2.3.IT/L2.3.IT_3.bed"

    # Load the BED file into a NumPy array
    bed_data = np.loadtxt(bed_file, dtype=str, delimiter='\t')

    f = open("/gpfs/slayman/pi/gerstein/asr94/senior_thesis_proj/output/nn/L2.3.IT/L2.3.IT_3_perm.txt", "w")
    # mse_list = []
    corr_dict = {}
    
    # j = 0

    random.seed(42)
    random_integers = random.sample(range(len(bed_data)), 1)
    for i in random_integers:
        gene = bed_data[i] # gene expression row
        y = np.array(gene[6:], dtype=np.double) # just the expression values across all samples for that gene
        x = find_cis_windows(int(gene[1]), vcf_array, vcf_columns) # will need to make this cis window of all the samples and snp dosages

        if (len(x) == 0): # if no snps for this gene
            continue
        
        input = []
        # for proper conversion into numpy array
        for entry in x:
            input.append(entry)

        x = np.array(input, dtype=np.int8)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        
        # create instance of model
        model = create_model(X_train)
  
        param_grid = {
            'model__hidden_layer_size': [50, 100],  # Hidden layer size
            'model__activation': ['relu', 'tanh'],  # Activation function
            'model__optimizer' : ['SGD', 'Adam']
        }   

        print("begin grid")

        # Perform grid Search
        keras_regressor = KerasRegressor(model, verbose=0)

        grid_search = GridSearchCV(estimator=keras_regressor, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        print("fitted grid")

        # Get best parameters and best score
        best_params = grid_search.best_params_

        print(f"Best Parameters: {best_params}")
        print('\n')

    f.close()

def create_model(X_train, activation='relu', hidden_layer_size=50, optimizer='adam'):
    # create an instance of the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=activation, input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(hidden_layer_size, activation=activation),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(1)  # No activation function in output layer for regression (want to predict numerical values directly, without transformation)
    ])

    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model


CIS_WINDOW = 1000000 # 1 million base pairs

def find_cis_windows(TSS, vcf_array, vcf_columns):
    start = TSS - CIS_WINDOW
    end = TSS + CIS_WINDOW

    snp_sample_mapping = {}
    for i in range(len(vcf_array)):
        snp_row = vcf_array[i]

        if start <= int(snp_row[1]) and int(snp_row[1]) <= end: # within cis window of this gene!
            # need to add all genotype dosages to dict, specific to each sample id.
            sample_genotypes = snp_row[9:]
            for j in range(len(sample_genotypes)):
                # get corresponding patient id to use for storing in dict. 
                sample_id = vcf_columns[j]
                if sample_genotypes[j] == '0/0':
                    dosage = 0
                elif sample_genotypes[j] == '0/1' or sample_genotypes[j] == '1/0':
                    dosage = 1
                elif sample_genotypes[j] == '1/1':
                    dosage = 2

                if sample_id in snp_sample_mapping.keys():
                    # append genotype dosage to existing list
                    snp_sample_mapping[sample_id].append(dosage)
                else: 
                    # create new entry
                    snp_sample_mapping[sample_id] = [dosage]
    
    return snp_sample_mapping.values()


if __name__ == "__main__":
    main()