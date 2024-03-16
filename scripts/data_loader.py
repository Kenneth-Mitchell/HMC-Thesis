from dask.distributed import Client
from image import Image
import os
import dask
from dask.diagnostics import ProgressBar
from deepforest import main
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask.delayed  # Import the delayed decorator

from deepforest import main
from deepforest import get_data
from deepforest import utilities
import matplotlib.pyplot as plt
import rasterio
import rasterio.features
import rasterio.warp
import os
import pandas as pd
import yaml
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Initialize R packages
neonUtilities = importr('neonUtilities')
base = importr('base')

# Load configuration and token
with open('../config.yml', 'r') as file:
    config = yaml.safe_load(file)

with open('../API_token.txt', 'r') as file:
    token = file.read()




@dask.delayed
def download_data_chunk(dpID, site, year):
    #given the dpID and site, download the data (either rgb or hsi)
    

    # basic call structure
    # neonUtilities.zipsByProduct(dpID=dpID, site=base.c(site), 
    #                                 savepath=savepath, package='basic',
    #                                 check_size='FALSE', token=token)
    pass

@dask.delayed 
def process_image(rgb_file_path, hsi_file_path, save_path):
    img = Image(rgb_file_path)
    img.generate_hsi_trees(hsi_file_path)
    img.save_hsi_trees(save_path) #TODO

def prepare_phenogeo():
    phenogeo_csv = "/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/phenogeo.csv"
    df = pd.read_csv(phenogeo_csv)
    df = pd.read_csv(phenogeo_csv).dropna(subset=['adjNorthing', 'adjEasting', 'uid'])
    df = df.drop_duplicates(subset='uid')
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    df = prepare_phenogeo()
    
    # Create a Dask CUDA cluster
    cluster = LocalCUDACluster()

    # Connect a Dask client to the cluster
    client = Client(cluster)

    # Log the dashboard link
    print(client.dashboard_link)

    rgb_dpID = config['dpID'][0]
    hsi_dpID = config['dpID'][1]
    with ProgressBar():
        tasks = []
        for batch in config['batches']:
            year = list(batch.items())[0][0]
            sites = list(batch.items())[0][1]
            for site in sites:
                #download rgb
                tasks.append(download_data_chunk(rgb_dpID, site, year))
                #download hsi
                tasks.append(download_data_chunk(hsi_dpID, site, year))
                #process images
                for root, dirs, files in #TODO: need the rgb and hsi paths
                    for file in files:
                        file_path = os.path.join(root, file)
                        tasks.append(process_image(rgb_file_path, hsi_file_path, save_path))


        # Compute all tasks
        results = dask.compute(*tasks)

if __name__ == '__main__':
    main()