from deepforest import main
from deepforest import get_data
from deepforest import utilities
import matplotlib.pyplot as plt
import rasterio
import rasterio.features
import rasterio.warp
import os
import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import numpy as np
import math 
import h5py


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
        self.gdf = pd.DataFrame()
        self.annotations = None
        self.deepforest_image = get_data(self.path)
    
    def plot(self):
        plt.imshow(self.data.transpose(1, 2, 0)) 

    def get_bounding_boxes(self, score_threshold=.5, patch_size=500, patch_overlap=0.25):
        # Predictions
        boxes = model.predict_tile(self.deepforest_image, patch_size= patch_size, patch_overlap=patch_overlap, return_plot=False)
        self.boxes = boxes
        self.gdf = utilities.annotations_to_shapefile(self.boxes, transform=self.transform, crs=self.crs)
        self.gdf = self.gdf[self.gdf['score'] > score_threshold]

    def annotate(self, ground_truth_df, threshold=10):
        if self.gdf.empty:
            self.get_bounding_boxes()
        self.gdf['ground_truth'] = None
        self.gdf['distance'] = None
        self.gdf['labeled'] = False
        
        df = ground_truth_df.copy()

        # Filter df to only include rows within utm coordinates of the ima
        utm_zone = self.gdf.crs.coordinate_operation.name[-3:]
        df = df[df['utmZone'] == utm_zone]
        df = df[(df['adjNorthing'] >= self.gdf.total_bounds[1]) & (df['adjNorthing'] <= self.gdf.total_bounds[3])]
        df = df[(df['adjEasting'] >= self.gdf.total_bounds[0]) & (df['adjEasting'] <= self.gdf.total_bounds[2])]

        # Convert the centroid coordinates to a numpy array
        centroid_array = np.array(list(zip(self.gdf.geometry.centroid.y, self.gdf.geometry.centroid.x)))

        for index, row in df.iterrows(): #TODO currently will override multiple competing values
            ground_truth_utm = pd.to_numeric(row[['adjNorthing', 'adjEasting']].values).reshape(1, -1)
            distances = cdist(ground_truth_utm, centroid_array)
            distance = min(distances[0, :])
            
            closest_index = np.argmin(distances, axis=1)
            if distance < threshold:
                self.gdf.loc[closest_index, 'ground_truth'] = row.to_frame().T # TODO Why nan?
                self.gdf.loc[closest_index, 'distance'] = distance
                self.gdf.loc[closest_index, 'labeled'] = True
                
    def generate_hsi_trees(self, hsi_path):
        self.get_hsi_points() #TODO only run if needed
        f = h5py.File(hsi_path, 'r')
        reflectance_data =f['GRSM']['Reflectance']['Reflectance_Data'][:]
        reflectance_data = np.rot90(reflectance_data, k=2, axes=(0,1))
        for index, row in self.gdf.iterrows():
            if row['labeled'] == False:
                continue
            print(row)
            xmin, ymin, xmax, ymax = row['xmin_hsi'], row['ymin_hsi'], row['xmax_hsi'], row['ymax_hsi']
            subset = reflectance_data.copy()[xmin:xmax, ymin:ymax, :]
            yield subset, row


    def get_hsi_points(self):
        if self.gdf.empty:
            self.get_bounding_boxes()
        
        self.gdf['xmin_hsi'] = self.gdf['xmin'].apply(lambda x: max(0,math.floor(x / 10)))
        self.gdf['ymin_hsi'] = self.gdf['ymin'].apply(lambda x: max(0,math.floor(x / 10)))
        self.gdf['xmax_hsi'] = self.gdf['xmax'].apply(lambda x: math.ceil(x / 10))
        self.gdf['ymax_hsi'] = self.gdf['ymax'].apply(lambda x: math.ceil(x / 10))
       
    
    def split_into_trees(self):
        if self.boxes.empty:
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
            