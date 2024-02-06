from dask.distributed import Client
from bbox import Image
import os
import dask
from dask.diagnostics import ProgressBar
from deepforest import main

@dask.delayed
def process_image(file_path):
    img = Image(file_path)
    img.split_into_trees()

def main():
    # Start a local Dask client
    client = Client() 

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