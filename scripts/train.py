from model import CNNSAM
import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
model = CNNSAM(in_channels=3, num_classes=1)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Print average training loss
        print(f”Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}“)
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # Print validation accuracy
        print(f”Epoch {epoch+1}, Validation Accuracy: {100*correct/total}%“)
# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)