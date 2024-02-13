import os
import pandas as pd
import glob

# Set the data path
input_path = 'input_data/phenology'
output_path = 'temp_data'

### Merge all the phenology data into one csv file ###

# create an empty dataframe to store all the merged dataframes
pheno_df = pd.DataFrame()

# loop through each directory in the path
for folder in os.listdir(input_path):
    folder_path = os.path.join(input_path, folder)
    if os.path.isdir(folder_path):
        # read the csv files
        
        for f in glob.glob(folder_path + '/*phe_perindividual.*'):
            perindividual_path = f

        for f in glob.glob(folder_path + '/*phe_statusintensity*.csv'):
            statusintensity_path = f

        perindividual_df = pd.read_csv(perindividual_path)
        statusintensity_df = pd.read_csv(statusintensity_path)
        # join the dataframes on individualID
        try: 
            merged_df = pd.merge(perindividual_df, statusintensity_df, on='individualID')
        except:
            continue
        # append the merged dataframe to the pheno_df dataframe
        pheno_df = pd.concat([pheno_df,merged_df])

# remove leading and trailing whitespaces from all the values in the dataframe
pheno_df = pheno_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# write the pheno_df dataframe to a new csv file
pheno_path = os.path.join(output_path, 'pheno.csv')
pheno_df.to_csv(pheno_path, index=False)

print(f"Phenology data merged and saved to {pheno_path}")

### Merge phenology and geolocation data ###
input_path='temp_data'

geo_path = os.path.join(input_path, 'geo.csv')
if not os.path.isfile(geo_path):
    raise FileNotFoundError(f"File '{geo_path}' does not exist. /n Please run geolocation.R first, using geolocate_rerun.")

geo_df = pd.read_csv(geo_path)

# merge the dataframes on individualID
merged_df = pd.merge(pheno_df, geo_df, on='individualID', how='outer')

# write the merged dataframe to a new csv file
merged_path = os.path.join(output_path, 'phenogeo.csv')
merged_df.to_csv(merged_path, index=False)

print(f"Phenology and geo data merged and saved to {merged_path}")