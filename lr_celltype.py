import numpy as np
import random
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def main():
    
    # Read the VCF (genotype variants) file into a pandas DataFrame
    vcf_data = pd.read_csv("/home/asr94/project/data/genotype_data/L2.3.IT/L2.3.IT_5.vcf", sep='\t', skiprows=1)
    vcf_columns = vcf_data.columns
    vcf_columns = vcf_columns[9:] # just the sample ids in a list

    vcf_array = vcf_data.values # converting to numpy array

    # Load the BED file into a NumPy array
    bed_data = np.loadtxt("/home/asr94/project/data/expr_data/L2.3.IT/L2.3.IT_5.bed", dtype=str, delimiter='\t')
    f = open("/gpfs/slayman/pi/gerstein/asr94/senior_thesis_proj/output/linreg/L2.3.IT/L2.3.IT_5.txt", "w")

    corr_dict = {}

    random.seed(42)
    random_integers = random.sample(range(len(bed_data)), 500)
    # make a linear regression model for 500 randomly selected genes in this cell type's expression file
    for i in range(len(bed_data)):
        if i in random_integers:
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
            model = LinearRegression().fit(X_train, y_train) # fitting linear regression model on data
            
            # Predict on the test data
            y_pred = model.predict(X_test)

            corr = stats.pearsonr(y_test, y_pred)
            corr_dict[gene[3]] = corr

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            f.write(gene[3] + ": MSE: " + str(mse) + " Pearson:  " + str(corr) + '\n')

    sorted_corrs = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
    # Printing the top five key-value pairs (ranking genes for GWAS based on correlation)
    for key, value in sorted_corrs[:5]:
        f.write(f"{key}: {value}")

    f.close()

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