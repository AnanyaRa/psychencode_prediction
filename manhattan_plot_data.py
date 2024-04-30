import numpy as np
from matplotlib import pyplot as plt

import random
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error

def main():
    # gwas associations for the umbrella trait "psychiatric disorders"
    data=pd.read_csv('/home/asr94/project/gwas-association-downloaded_2024-04-11-EFO_0005223-withChildTraits.tsv',sep='\t')
    psychiatric_info = data[['CHR_ID', 'CHR_POS', 'P-VALUE']]

    gwas_tuples = psychiatric_info['CHR_POS'].values

    # first, get list of all the SNP positions in top performing genes from GWAS exploration sheet. 
    # merge cis windows for each into one giant list. 

    # some of the chromosome IDs are just X, so those would be ignored in this comparison. 

    # then, take that as set and overlap with psychiatric_snps...
    # actually, make tuples of chromosome id and chromosome pisition across thw GWAS and my top performers. 
    # take the overlap of this...
    # and get the p values of those. 

    # Read the VCF (genotype variants) file into a pandas DataFrame
    vcf_data = pd.read_csv("/home/asr94/project/data/genotype_data/L2.3.IT/L2.3.IT_22.vcf", sep='\t', skiprows=1)
    vcf_columns = vcf_data.columns
    vcf_columns = vcf_columns[9:] # just the sample ids in a list
    vcf_array = vcf_data.values # converting to numpy array

    # Load the BED file into a NumPy array
    bed_data = np.loadtxt("data/expr_data/L2.3.IT/L2.3.IT_22.bed", dtype=str, delimiter='\t')

    gene_compare = "KANSL1"
    sample_tuples = [] # list of tuples containing (chr_id, chr_pos) for each SNP within each cis window of the 5 top performing genes
    for i in range(len(bed_data)):
        gene = bed_data[i] # gene expression row
        if gene[3] == gene_compare:
            sample_tuples = find_cis_windows(int(gene[1]), vcf_array) # will need to make this cis window of all the samples and snp dosages

    # # Convert lists to sets
    set1 = set(gwas_tuples)
    set2 = set(sample_tuples)

    # Calculate overlap
    overlap = set1.intersection(set2)
    # print("Overlap:", overlap)

    # Filter rows where column 'CHR_POS' has values in the overlap set
    filtered_df = psychiatric_info[psychiatric_info['CHR_POS'].isin(overlap)]
    pd.set_option('display.max_rows', None)  # Set to None to display all rows
    pd.set_option('display.max_columns', None)  # Set to None to display all columns
    
    def format_value(value):
        if isinstance(value, float):
            return '{:.15f}'.format(value)
        else:
            return str(value)

    # Join each row with commas and formatted numeric values
    result = [', '.join(map(format_value, row)) for row in filtered_df.values]

    # Print the result
    for row in result:
        print("add_row("+row+")")

    print("Number of elements in overlap:", len(overlap))

CIS_WINDOW = 1000000 # 1 million base pairs

def find_cis_windows(TSS, vcf_array):
    start = TSS - CIS_WINDOW
    end = TSS + CIS_WINDOW

    snp_positions = []
    for i in range(len(vcf_array)):
        snp_row = vcf_array[i]

        if start <= int(snp_row[1]) and int(snp_row[1]) <= end: # within cis window of this gene!
            snp_positions.append(str(snp_row[1]))

    return snp_positions

if __name__ == "__main__":
    main()