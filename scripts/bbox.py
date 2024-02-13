from deepforest import main
from deepforest import get_data
from deepforest import utilities
import matplotlib.pyplot as plt
import rasterio
import rasterio.features
import rasterio.warp
import os
import torch
torch.set_float32_matmul_precision('medium')
model = main.deepforest()
model.use_release()

class Image():
    def __init__(self, path):
        self.path = path
        self.image = rasterio.open(path)
        self.bounds = self.image.bounds
        self.crs = self.image.crs
        self.transform = self.image.transform
        self.data = self.image.read()
        self.shape = self.data.shape
        self.boxes = None
        self.gdf = None
        self.annotations = None
        self.deepforest_image = get_data(self.path)
    
    def plot(self):
        plt.imshow(self.data.transpose(1, 2, 0)) 

    def get_bounding_boxes(self):
        # Predictions
        boxes = model.predict_tile(self.deepforest_image, patch_size=500,patch_overlap=0.25)
        self.boxes = boxes
        self.gdf = utilities.annotations_to_shapefile(self.boxes, transform=self.transform, crs=self.crs)

    def annotate(self, vst_path):
        if not self.gdf:
            self.get_bounding_boxes()
        
        

    def split_into_trees(self):
        if not self.boxes:
            self.get_bounding_boxes()
        output_dir = f"bounding_box_images/{self.path.split('/')[-1].split('.')[0]}"
        os.makedirs(output_dir, exist_ok=True)

        # Sort the dataframe by score and filter to top 500
        temp = self.gdf.copy()
        temp =  temp.sort_values(by="score", ascending=False)
        temp = temp.head(10)


        # Open the original image
        with rasterio.open(self.path) as src:
            # Loop through each bounding box in the gdf dataframe
            for index, row in temp.iterrows():
                # Get the score of the bounding box
                score = row['score']
                # Get the bounding box coordinates
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

                # Create a rasterio window for the bounding box
                window=rasterio.windows.Window(xmin, ymin, xmax - xmin, ymax - ymin)

                # Read the subset of the image within the bounding box
                subset = src.read(window=window)

                # Create a new rasterio dataset for the subset
                subset_dataset = rasterio.open(
                    f"{output_dir}/bounding_box_{index}_score={score}.tif",
                    'w',
                    driver='GTiff',
                    height=subset.shape[1],
                    width=subset.shape[2],
                    count=subset.shape[0],
                    dtype=subset.dtype,
                    crs=src.crs,
                    transform=src.window_transform(window)
                )
                
                # Write the subset to the new dataset
                subset_dataset.write(subset)
                subset_dataset.close()
            