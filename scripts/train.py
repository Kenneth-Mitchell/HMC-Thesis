import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cleaning
df = pd.read_csv('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/hsi_tensors/master.csv')
df = df.drop(['Unnamed: 0.1'], axis=1)
df = df.drop_duplicates()
df = df.drop(index=36)
df.dropna(subset=['taxonID_y'], inplace=True)
df['distance'] = df['distance'].astype(float)
df = df[df['taxonID_y'].map(df['taxonID_y'].value_counts()) >= 50]
df = df.loc[df.groupby('geometry')['distance'].idxmin()]
#TODO use a better filtering method
df = df[df['taxonID_y'].map(df['taxonID_y'].value_counts()) >= 20]

# Define hyperparameters
batch_size = 32
learning_rate = 0.01
num_epochs = 100
taxonID_to_int = {taxonID: 0 if taxonID != 'LITU' else 1 for taxonID in df['taxonID_y'].unique()}
int_to_taxonID = {0: 'OTHER', 1: 'LITU'}
class ResizeToSize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, data):
        data = data.float()
        data = nn.functional.interpolate(data.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False)
        return data.squeeze(0).squeeze(0)

class CustomDataset(Dataset):
    def __init__(self, dataframe, target_size=(10, 10, 426), augment=True):
        self.dataframe = dataframe
        self.transform = ResizeToSize(target_size)
        self.augment = augment

        # Define augmentation transformations
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),

        ])

    def __len__(self):
        return len(self.dataframe)

    def min_max_scale(self, data):
        # Perform min-max scaling on the data
        min_val = data.min()
        max_val = data.max()
        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data

    def zscore_scale(self, data):
        # Perform z-score scaling on the data
        mean = data.mean()
        std = data.std()
        scaled_data = (data - mean) / std
        return scaled_data
    
    def zeroaware_scale(self, data):
        # Perform zero-aware scaling on the data
        nonzero_data = data[data != 0]
        mean = nonzero_data.mean()
        std = nonzero_data.std()
        scaled_data = data.clone()
        scaled_data[scaled_data != 0] = (nonzero_data - mean) / std
        np.clip(scaled_data, -1, 1)
        return scaled_data

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = f"/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/hsi_tensors/{row['siteID']}_{row['date']}_{row['adjEasting']}_{row['adjNorthing']}.h5"
        
        # Open the h5 file
        file = h5py.File(img_name, 'r')
        data = torch.from_numpy(file['subset'][:])
        file.close()

        # Apply resizing to the data
        data = self.transform(data)

        # Perform min-max scaling on the data
        # data = self.min_max_scale(data)
        # data = self.zscore_scale(data)
        # data = self.zeroaware_scale(data)

        data = data + 1

        # Swap the channel and x dimensions
        data = data.permute(2, 1, 0)

        label = torch.tensor(taxonID_to_int[row['taxonID_y']])

        # Apply augmentation if enabled
        if self.augment:
            data = self.augmentation(data)
        return data, label.float()



train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['taxonID_y'].map(taxonID_to_int))

train_dataset = CustomDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset(test_df)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# print(train_df['taxonID_y'].value_counts())
# print(sum([1 for x in train_df['taxonID_y'] if not x == 'LITU']))

model = CNNSAM(in_channels=426, num_classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    label_count = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        #strip outputs
        outputs = outputs.squeeze(1)
        # Compute binary cross-entropy loss
        loss = criterion(outputs, labels.float())  
        # print(outputs)
        loss.backward()
        optimizer.step()
        # print(outputs)

        running_loss += loss.item()
        # print(outputs)
        # Convert logits to predictions by thresholding
        predictions = (outputs > 0.5).float()  

        # Update correct predictions
        correct += (predictions == labels).sum().item() 
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total
    
    # Validation loop
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    batch_count = 0  # Track the number of batches
    confusion_matrix = torch.zeros(2, 2)
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            label_count += labels.sum()
            outputs = model(images)
            outputs = outputs.squeeze(1)
            # Compute binary cross-entropy loss directly from logits for validation
            val_loss += criterion(outputs, labels.float()).item() 
            
            # Convert logits to predictions by thresholding
            predictions = (outputs > 0.5).float()  
            val_correct += (predictions == labels).sum().item() 
            val_total += labels.size(0)
            batch_count += 1

            for t, p in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        print(confusion_matrix)

    # Compute average validation loss and accuracy
    val_loss /= batch_count
    val_accuracy = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

print(val_losses)
print(val_accuracies)
# plot loss with epochs
plt.plot(val_losses)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs. Epoch')
plt.savefig(f"/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/epochs/val_loss.png")
plt.close()
# plot accuracy with epochs
plt.plot(val_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. Epoch')
plt.savefig(f"/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/epochs/val_accuracy.png")
plt.close()

confusion_matrix = confusion_matrix.numpy()
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.plot()
plt.savefig(f"/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/epochs/confusion_matrix_epoch_{epoch}.png")
plt.close()
