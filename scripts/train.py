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

# cleaning
df = pd.read_csv('/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/hsi_tensors/master.csv')
df = df.drop(['Unnamed: 0.1'], axis=1)
df = df.drop_duplicates()
df = df.drop(index=36)
df.dropna(subset=['taxonID_y'], inplace=True)
df_min_distance = df.groupby('geometry')['distance'].min().reset_index()
df = df.merge(df_min_distance, on=['geometry', 'distance'], how='inner')
df = df[df['taxonID_y'].map(df['taxonID_y'].value_counts()) >= 50]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['taxonID_y'])

# Define hyperparameters
batch_size = 32
learning_rate = 0.5
num_epochs = 10
taxonID_to_int = {taxonID: 0 if taxonID != 'LITU' else 1 for taxonID in train_df['taxonID_y'].unique()}
int_to_taxonID = {0: 'OTHER', 1: 'LITU'}

class ResizeToSize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, data):
        data = data.float()
        data = nn.functional.interpolate(data.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False)
        return data.squeeze(0).squeeze(0)

class CustomDataset(Dataset):
    def __init__(self, dataframe, target_size=(10, 10, 426)):
        self.dataframe = dataframe
        self.transform = ResizeToSize(target_size)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = f"/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/hsi_tensors/{row['siteID']}_{row['date']}_{row['adjEasting']}_{row['adjNorthing']}.h5"
        # Open the h5 file
        file = h5py.File(img_name, 'r')

        data = torch.from_numpy(file['subset'][:])

        # Close the file
        file.close()
        label = torch.tensor(taxonID_to_int[row['taxonID_y']])
        # Apply resizing to the data
        data = self.transform(data)
        # Swap the channel and x dimensions
        data = data.permute(2, 1, 0)

        return data, label




train_dataset = CustomDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = CNNSAM(in_channels=426, num_classes=len(int_to_taxonID))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
val_dataset = CustomDataset(test_df)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        labels_one_hot = torch.zeros(labels.size(0), 2)  # Create a one-hot encoded tensor for binary classification
        labels_one_hot[range(labels.size(0)), labels.long()] = 1  # Set the corresponding class index to 1
        loss = criterion(outputs, labels_one_hot)  # Use one-hot encoded labels for binary cross-entropy loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total

    # Validation loop
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    batch_count = 0  # Track the number of batches
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            batch_size = labels.size(0)  # Get the actual batch size
            labels_one_hot = torch.zeros(batch_size, 2)  # Create a one-hot encoded tensor for binary classification
            labels_one_hot[range(batch_size), labels.long()] = 1  # Set the corresponding class index to 1
            val_loss += criterion(outputs, labels_one_hot).item()  # Use the same one-hot encoded labels for validation
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += batch_size
            batch_count += 1

    # Compute average validation loss and accuracy
    val_loss /= batch_count
    val_accuracy = val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
