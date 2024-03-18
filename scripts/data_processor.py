
from image import Image
from data_loader import prepare_phenogeo
import os
import dask
from dask.diagnostics import ProgressBar
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask.delayed  
import matplotlib.pyplot as plt
import rasterio
import rasterio.features
import rasterio.warp
import os
import pandas as pd

# rgb sample file /mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/DP3.30010.001/neon-aop-products/2021/FullSite/D07/2021_GRSM_5/L3/Camera/Mosaic/2021_GRSM_5_276000_3951000_image.tif
# hsi sample file /mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/DP3.30006.001/neon-aop-products/2021/FullSite/D07/2021_GRSM_5/L3/Spectrometer/Reflectance/NEON_D07_GRSM_DP3_282000_3957000_reflectance.h5



def make_image_dict(rgb, hsi):
    final_dict = {}
    hsi_dict = {}
    for root, dirs, files in os.walk(hsi):
        for file in files:
            if file.endswith('.h5'):
                hsi_file = os.path.join(root, file)
                hsi_year = hsi_file.split('/')[11]
                hsi_parts = os.path.basename(hsi_file).split('_')
                hsi_key = hsi_parts[2], hsi_year, hsi_parts[4], hsi_parts[5] # site, year, easting, northing
                hsi_dict[hsi_key] = hsi_file

    for root, dirs, files in os.walk(rgb):
        for file in files:
            if file.endswith('.tif'):
                rgb_file = os.path.join(root, file)
                rgb_parts = os.path.basename(rgb_file).split('_')
                rgb_key = rgb_parts[1], rgb_parts[0], rgb_parts[3], rgb_parts[4]# site, year, easting, northing
                if rgb_key in hsi_dict:
                    final_dict[rgb_key] = (rgb_file, hsi_dict[rgb_key])
    return final_dict

@dask.delayed 
def process_image(rgb_file, hsi_file):
    img = Image(rgb_file)
    for subset, row in img.generate_hsi_trees(hsi_file):

        rasterio.imwrite('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/hsi_tensors/' + row['Eas'] + row, subset)
        


def main():
    rgb = '/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/DP3.30010.001/neon-aop-products'
    hsi = '/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/DP3.30006.001/neon-aop-products'
    files = make_image_dict(rgb, hsi)
    # Create a Dask CUDA cluster
    cluster = LocalCUDACluster()

    # Connect a Dask client to the cluster
    client = Client(cluster)

    # Log the dashboard link
    print(client.dashboard_link)
    with ProgressBar():
        tasks = []
        for file in files.values():
            rgb_file = file[0]
            hsi_file = file[1]

            
            tasks.append(process_image(rgb_file, hsi_file))
            break

        # Compute all tasks
        results = dask.compute(*tasks)



    # hsi_image_path = "/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/hsi/NEON_D07_GRSM_DP3_275000_3951000_reflectance.h5"
    # img = Image(rgb_image_path)
    # # img.annotate(df)
    # # img.get_bounding_boxes()

    # img.generate_hsi_trees("/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/hsi/NEON_D07_GRSM_DP3_275000_3951000_reflectance.h5")
    # print(img.gdf.head())

if __name__ == '__main__':
    main()