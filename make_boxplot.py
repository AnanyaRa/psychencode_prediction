import numpy as np
from matplotlib import pyplot as plt

import random
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm.keras import TqdmCallback
import pandas as pd
from scipy.stats import wilcoxon

#make rf and nn for same data, get the pearson correlation values in a list for both and then plot them side by side
def main():
    
    # Read the VCF (genotype variants) file into a pandas DataFrame
    vcf_data = pd.read_csv("/home/asr94/project/data/genotype_data/Oligo/Oligo_10.vcf", sep='\t', skiprows=1)
    vcf_columns = vcf_data.columns
    vcf_columns = vcf_columns[9:] # just the sample ids in a list

    vcf_array = vcf_data.values # converting to numpy array

    # Load the BED file into a NumPy array
    bed_data = np.loadtxt("/home/asr94/project/data/expr_data/Oligo/Oligo_10.bed", dtype=str, delimiter='\t')
    # f = open("/gpfs/slayman/pi/gerstein/asr94/senior_thesis_proj/output/model_comparisons.txt", "w")

    # corr_dict = {}
    corr_lr_list = []
    corr_rf_list = []
    corr_nn_list = []
    random.seed(42)
    # make a linear regression, random forest, and neural network model for every gene in this cell type's expression file
    for i in range(len(bed_data)):
        gene = bed_data[i] # gene expression row
        y = np.array(gene[6:], dtype=np.double) # just the expression values across all samples for that gene

        x = find_cis_windows(int(gene[1]), vcf_array, vcf_columns) # will need to make this cis window of all the samples and snp dosages
        # print(vcf_columns)
        if (len(x) == 0): # if no snps for this gene
            continue
        
        input = []
        # for proper conversion into numpy array
        for entry in x:
            input.append(entry)

        x = np.array(input, dtype=np.int8)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        lr_model = LinearRegression().fit(X_train, y_train) # fitting linear regression model on data
            
        # Predict on the test data
        y_pred = lr_model.predict(X_test)

        lr_corr, _ = stats.pearsonr(y_test, y_pred)
        corr_lr_list.append(lr_corr)

        # create an instance of the Random Forest regression model
        rf_model = RandomForestRegressor(n_estimators=100, max_features='sqrt', bootstrap=True)

        # train the model on the training data
        rf_model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = rf_model.predict(X_test)

        rf_corr, _ = stats.pearsonr(y_test, y_pred)
        corr_rf_list.append(rf_corr)
        # corr_dict[gene[3]] = corr

        # create an instance of the Neural network model
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)  # No activation function in output layer for regression (want to predict numerical values directly, without transformation)
        ])

        nn_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # train the model on the training data
        nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[TqdmCallback(verbose=1)])

        # Predict on the test data
        y_pred = nn_model.predict(X_test)
        y_pred_reshaped = np.reshape(y_pred, (-1, 1))
        # Calculate correlation coefficient
        nn_corr = np.corrcoef(y_test, y_pred_reshaped, rowvar=False)[0, 1]
        corr_nn_list.append(nn_corr)


    # Set the figure size
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Pandas dataframe
    # for element in corr_nn_list:
    #     f.write(str(element))
    #     f.write("\n")
    # f.write("\n")
    # for element in corr_rf_list:
    #     f.write(str(element))
    #     # f.write(corr_rf_list)
    #     f.write("\n")
    # f.write("\n")
    # for element in corr_lr_list:
    #     # f.write(corr_lr_list)
    #     f.write(str(element))
    #     f.write("\n")

    data = pd.DataFrame({"NN": corr_nn_list, "RF": corr_rf_list, "LR": corr_lr_list})
    corr = wilcoxon(corr_nn_list, corr_rf_list)
    # print(corr.pvalue)
    # Plot the dataframe
    ax = data[['NN', 'RF', 'LR']].plot(kind='box', title='NN vs RF vs LR correlations (NN vs RF: ' + str(corr.pvalue) + ')')


    # Display the plot
    # plt.show()
    plt.savefig(f"/home/asr94/project/output/images/comparison/Oligo_10_boxplot.png")
    plt.close()

    # f.close()

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