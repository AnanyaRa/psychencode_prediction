import numpy as np
import random
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import tensorflow as tf
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt

def main():
    # Specify the path to your VCF file
    vcf_file = "/home/asr94/project/data/genotype_data/L2.3.IT/L2.3.IT_2.vcf"

    # # Read the VCF file into a pandas DataFrame
    vcf_data = pd.read_csv(vcf_file, sep='\t', skiprows=1)
    vcf_columns = vcf_data.columns
    vcf_columns = vcf_columns[9:] # just the sample ids in a list

    vcf_array = vcf_data.values # converting to numpy array

    # Specify the path to your BED file
    # "/home/asr94/project/data/expr_data/chandelier/chandelier_12.bed"
    bed_file = "/home/asr94/project/data/expr_data/L2.3.IT/L2.3.IT_2.bed"

    # Load the BED file into a NumPy array
    bed_data = np.loadtxt(bed_file, dtype=str, delimiter='\t')

    covariates_file = "/home/asr94/project/data/cov_data/L2.3.IT_wo_ROSMAP.cov.20_expr_PCs_wo_ROSMAP.bed"
    covariates_data = pd.read_csv(covariates_file, delimiter='\t')
    covariate_cols = covariates_data.columns[1:] # just the sample IDs, without 'SampleID' label
    covariate_array = covariates_data.values # converting to numpy array

    f = open("/gpfs/slayman/pi/gerstein/asr94/senior_thesis_proj/output/nn/L2.3.IT/cov_L2.3.IT_2.txt", "w")
    # mse_list = []
    corr_dict = {}
    
    j = 0

    random.seed(42)
    random_integers = random.sample(range(len(bed_data)), 500)
    for i in range(len(bed_data)):
        if i in random_integers:
            gene = bed_data[i] # gene expression row
            y = np.array(gene[6:], dtype=np.double) # just the expression values across all samples for that gene
            snp_sample_mappings = find_cis_windows(int(gene[1]), vcf_array, vcf_columns) # will need to make this cis window of all the samples and snp dosages
            x = add_covariates(snp_sample_mappings, covariate_array, covariate_cols)
            if (len(x) == 0): # if no snps for this gene
                continue
            
            input = []
            # for proper conversion into numpy array
            for entry in x:
                input.append(entry)

            x = np.array(input, dtype=np.int8)

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
            # create an instance of the Neural network model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1)  # No activation function in output layer for regression (want to predict numerical values directly, without transformation)
            ])

            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            # train the model on the training data
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[TqdmCallback(verbose=1)])

            # Predict on the test data
            y_pred = model.predict(X_test)
            y_pred_reshaped = np.reshape(y_pred, (-1, 1))
            # Calculate correlation coefficient
            corr = np.corrcoef(y_test, y_pred_reshaped, rowvar=False)[0, 1]

            corr_dict[gene[3]] = corr

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            f.write(gene[3] + ": MSE: " + str(mse) + " Correlation:  " + str(corr) + '\n')
        

    sorted_corrs = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
    # Printing the top five key-value pairs (ranking genes for GWAS based on correlation)
    f.write('\n')
    for key, value in sorted_corrs[:5]:
        f.write(f"{key}: {value}")
        f.write('\n')

    f.close()

def add_covariates(snp_sample_mapping, covariate_array, covariate_cols):
    for i in range(len(covariate_array)):
        covariate_row = covariate_array[i]
        sample_encodings = covariate_array[1:] # just the one hot encodings for this given covariate!
        for j in range(len(sample_encodings)):
            sample_id = covariate_cols[j]
            if sample_id in snp_sample_mapping.keys():
                # append covariate encoding to existing list for this sample_id
                snp_sample_mapping[sample_id].append(sample_encodings[j])
            else: 
                # create new entry
                snp_sample_mapping[sample_id] = [sample_encodings[j]]
    
    print(snp_sample_mapping)
    return snp_sample_mapping.values()

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
    
    # add to the values the encodings for covariates. so that's it is inputted along w the dosages
    return snp_sample_mapping

if __name__ == "__main__":
    main()