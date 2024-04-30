import numpy as np
import random
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

def main():
    
    # Read the VCF (genotype variants) file into a pandas DataFrame
    vcf_data = pd.read_csv("/home/asr94/project/data/genotype_data/L2.3.IT/L2.3.IT_22.vcf", sep='\t', skiprows=1)
    vcf_columns = vcf_data.columns
    vcf_columns = vcf_columns[9:] # just the sample ids in a list
    # variant_names = vcf_data['POS']
    # print(variant_names)

    vcf_array = vcf_data.values # converting to numpy array

    # Load the BED file into a NumPy array
    bed_data = np.loadtxt("/home/asr94/project/data/expr_data/L2.3.IT/L2.3.IT_22.bed", dtype=str, delimiter='\t')
    covariates_file = "/home/asr94/project/data/cov_data/L2.3.IT_wo_ROSMAP.cov.20_expr_PCs_wo_ROSMAP.bed"
    covariates_data = pd.read_csv(covariates_file, delimiter='\t')
    covariate_cols = covariates_data.columns[1:] # just the sample IDs, without 'SampleID' label
    covariate_array = covariates_data.values # converting to numpy array
    f = open("/gpfs/slayman/pi/gerstein/asr94/senior_thesis_proj/output/rf/covariates/L2.3.IT/L_22.txt", "w")

    corr_dict = {}
    # make a random forest model for every gene in this cell type's expression file
    for i in range(len(bed_data)):
        gene = bed_data[i] # gene expression row
        y = np.array(gene[6:], dtype=np.double) # just the expression values across all samples for that gene

        TSS = int(gene[1])
        snp_sample_mappings, variant_pos = find_cis_windows(TSS, vcf_array, vcf_columns) # will need to make this cis window of all the samples and snp dosages
        x, variant_pos = add_covariates(snp_sample_mappings, covariate_array, covariate_cols, variant_pos)

        if (len(x) == 0): # if no snps for this gene
            continue
        
        input = []
        # for proper conversion into numpy array
        for entry in x:
            input.append(entry)

        x = np.array(input, dtype=np.int8)
        # columns should be each variant name
        X = pd.DataFrame(x, columns=variant_pos)
        X.columns = X.columns.astype(str)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # create an instance of the Random Forest regression model
        model = RandomForestRegressor(n_estimators=100, max_features='sqrt', bootstrap=True)

        # train the model on the training data
        model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test)

        corr = stats.pearsonr(y_test, y_pred)
        corr_dict[gene[3]] = corr

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        f.write(gene[3] + ": MSE: " + str(mse) + " Pearson:  " + str(corr) + '\n')

    sorted_corrs = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
    # Printing the top five key-value pairs (ranking genes for GWAS based on correlation)
    f.write('\n')
    for key, value in sorted_corrs[:5]:
        f.write(f"{key}: {value}")
        f.write('\n')

    f.close()


def add_covariates(snp_sample_mapping, covariate_array, covariate_cols, feature_names):
    for i in range(len(covariate_array)):
        covariate_row = covariate_array[i]
        covariate_name = covariate_row[0]
        feature_names.append(covariate_name)
        sample_encodings = covariate_row[1:] # just the one hot encodings for this given covariate!
        for j in range(len(sample_encodings)):
            sample_id = covariate_cols[j]
            if sample_id in snp_sample_mapping.keys():
                # append covariate encoding to existing list for this sample_id
                snp_sample_mapping[sample_id].append(sample_encodings[j])
            else: 
                # create new entry
                snp_sample_mapping[sample_id] = [sample_encodings[j]]
    
    # print(snp_sample_mapping)
    return snp_sample_mapping.values(), feature_names

CIS_WINDOW = 1000000 # 1 million base pairs

def find_cis_windows(TSS, vcf_array, vcf_columns):
    start = TSS - CIS_WINDOW
    end = TSS + CIS_WINDOW

    snp_sample_mapping = {}
    variant_pos = []
    for i in range(len(vcf_array)):
        snp_row = vcf_array[i]

        if start <= int(snp_row[1]) and int(snp_row[1]) <= end: # within cis window of this gene!
            # need to add all genotype dosages to dict, specific to each sample id.
            variant_pos.append(snp_row[1])
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

    return snp_sample_mapping, variant_pos

if __name__ == "__main__":
    main()