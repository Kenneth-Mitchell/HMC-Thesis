import os
import yaml
import dask
import dask.bag as db
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import dask.bag as db

# Initialize R packages
neonUtilities = importr('neonUtilities')
base = importr('base')

# Load configuration and token
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

with open('API_token.txt', 'r') as file:
    token = file.read()

# Function to download data from NEON API

# Function to download a specific chunk of data
@dask.delayed
def download_data_chunk(dpID, site, savepath):
    try:
        neonUtilities.zipsByProduct(dpID=dpID, site=base.c(site), 
                                    savepath=savepath, package='basic',
                                    check_size='FALSE', token=token)
        
        # neonUtilities.stackByTable(filepath=download_path)
        return [os.path.join(savepath, f) for f in os.listdir(savepath) if os.path.isfile(os.path.join(savepath, f))]
    except Exception as e:
        print(f"Error downloading data chunk: {e}")
        return []


# Main script
def main():
    for chunk in config['chunks']:
        # Create save path
        savepath = os.path.join(config['savepath'], chunk)
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        # Download data
        tasks = download_data_chunk()

        results = dask.compute(*tasks)

        train(results)


if __name__ == "__main__":
    main()
