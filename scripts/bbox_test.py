from dask.distributed import Client
from bbox import Image
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

@dask.delayed  # Decorate the function to make it delayed
def process_image(file_path):
    img = Image(file_path)
    img.split_into_trees()

def main():
    # Create a Dask CUDA cluster
    cluster = LocalCUDACluster()

    # Connect a Dask client to the cluster
    client = Client(cluster)

    # Log the dashboard link
    print(client.dashboard_link)
    with ProgressBar():
        tasks = []
        for root, dirs, files in os.walk("/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/rgb"):
            for file in files:
                file_path = os.path.join(root, file)
                tasks.append(process_image(file_path))

        # Compute all tasks
        results = dask.compute(*tasks)

if __name__ == '__main__':
    main()