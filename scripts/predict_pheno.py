import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
import sys
from model import CNNSAM
import numpy as np
from image import Image

sample_image_path = "/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/rgb_test/2021_GRSM_5_275000_3951000_image.tif"
sample_HSI_path = "/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/hsi/NEON_D07_GRSM_DP3_275000_3951000_reflectance.h5"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNSAM(426, 1).to(device)
model.load_state_dict(torch.load('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/pheno_model.pth'))
model.eval()

class ResizeToSize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, data):
        data = data.float()
        data = nn.functional.interpolate(data.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False)
        return data.squeeze(0).squeeze(0)
    
# Load the data
class CustomDataset(Dataset):
    def __init__(self, dataframe, target_size=(10, 10, 426)):
        self.transform = ResizeToSize(target_size)
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def min_max_scale(self, data):
        # Perform min-max scaling on the data
        min_val = data.min()
        max_val = data.max()
        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = f"/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/unknown_hsi_tensors/{row['file']}"
        
        # Open the h5 file
        file = h5py.File(img_name, 'r')
        data = torch.from_numpy(file['subset'][:])
        file.close()

        # Apply resizing to the data
        data = self.transform(data)

        # Perform min-max scaling on the data
        data = self.min_max_scale(data)

        # Swap the channel and x dimensions
        data = data.permute(2, 1, 0)
        return data, row['file']

# Locate trees in the image
master = []
img = Image(sample_image_path)
# img.get_bounding_boxes()
# for subset, row in img.generate_hsi_trees(sample_HSI_path, all=True):
#         with h5py.File(f"/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/unknown_hsi_tensors/{row['xmin']}_{row['ymin']}_{row['xmax']}_{row['ymax']}.h5", 'w') as f:
#             f.create_dataset('subset', data=np.array(subset))
#             f.close()
#         # master.append(f"{row['xmin']}_{row['ymin']}_{row['xmax']}_{row['ymax']}")

# for file in os.walk('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/unknown_hsi_tensors/'):
#     for f in file[2]:
#         if f.endswith('.h5'):
#             master.append(f)

# master = pd.DataFrame(master, columns=['file'])
# master.to_csv('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/unknown_hsi_tensors/master.csv', index=False)

# Load the data
dataframe = pd.read_csv('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/unknown_hsi_tensors/master.csv')
dataframe = dataframe[dataframe['predicted_class'] == 1]
data = CustomDataset(dataframe)
data_loader = DataLoader(data, batch_size=1)

predictions = []
for images, file in data_loader:
    images = images.to(device)
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        predicted_class_pheno = (outputs > 0.5)
        predictions.append((file[0], int(predicted_class_pheno.item())))
    

master = pd.merge(dataframe, pd.DataFrame(predictions, columns=['file', 'predicted_class_pheno']), on='file')
master.to_csv('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/unknown_hsi_tensors/master.csv', index=False)
master = pd.read_csv('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/unknown_hsi_tensors/master.csv')
rgb_img = img.data.copy().transpose(1,2,0)
rgb_img_2 = img.data.copy().transpose(1,2,0)
#draw boxes on tulip trees
centroids = []
for file in master[master['predicted_class_pheno'] == 1]['file']:
    # Open the test image
    
    xmin, ymin, xmax, ymax = file.split('_')
    ymax = ymax.split('.')[0]
    rgb_points = int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))
    rgb_img[rgb_points[0]-2:rgb_points[0], rgb_points[1]:rgb_points[3], :] = [255, 100, 0]
    rgb_img[rgb_points[2]:rgb_points[2]+2, rgb_points[1]:rgb_points[3], :] = [255, 100, 0]
    rgb_img[rgb_points[0]:rgb_points[2], rgb_points[1]-2:rgb_points[1], :] = [255, 100, 0]
    rgb_img[rgb_points[0]:rgb_points[2], rgb_points[3]:rgb_points[3]+2, :] = [255, 100, 0]

    

    centroid = rgb_points[0] + (rgb_points[2] - rgb_points[0]) / 2, rgb_points[1] + (rgb_points[3] - rgb_points[1]) / 2
    centroids.append(centroid)

    # print(rgb_points)
    # print(rgb_img[rgb_points[0]:rgb_points[2], rgb_points[1]:rgb_points[3], :].shape)
    # print(rgb_img[rgb_points[0]:rgb_points[2], rgb_points[1]:rgb_points[3], :])
    

    # plt.imsave('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/rgb.png',rgb_img[rgb_points[0]:rgb_points[2], rgb_points[1]:rgb_points[3], :].astype(np.uint8))
    # with h5py.File(f"/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/unknown_hsi_tensors/{file}", 'r') as f:
    #     hsi_img = f['subset'][:].copy()[:,:,[58, 34, 19]]
    #     hsi_img = hsi_img / hsi_img.max()
    #     f.close()
    # plt.imsave('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/hsi.png',hsi_img)

    # break

centroids = np.array(centroids)
plt.scatter(centroids[:,1], centroids[:,0], c='r', s=.5, alpha=0.5)
plt.imshow(rgb_img_2)
plt.axis('off')
plt.savefig('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/figure_final_pheno.png',bbox_inches='tight')
plt.close()

# Save the modified image
plt.imsave('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/figure_pheno_class.png', rgb_img)



