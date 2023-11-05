import numpy as np
import pandas as pd
import os
import pickle
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch
import torchvision
from torchvision.models import ResNet50_Weights, DenseNet121_Weights
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from itertools import chain
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import resample
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    multilabel_confusion_matrix, roc_curve, auc, classification_report
from collections import Counter
from math import ceil

# Device configuration GPU support for MAC
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("MPS device not found.")

# Paths to Images and DataEntry file
all_xray_df = pd.read_csv('NihXrayData/Data_Entry_2017_v2020.csv')
allImagesGlob = glob('NihXrayData/images*/images/*.png')
# eof

all_image_paths = {os.path.basename(x): x for x in
                   allImagesGlob}
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df = all_xray_df[all_xray_df['Finding Labels'] != 'No Finding']
all_xray_df.sample(3)

# # Data Pre Processing ####
condition_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                    'Effusion', 'Pneumonia', 'Pleural_Thickening',
                    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
for label in condition_labels:
    all_xray_df[label] = all_xray_df['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
all_xray_df.head(20)

all_xray_df['disease_vec'] = all_xray_df.apply(lambda target: [target[condition_labels].values], 1).map(
    lambda target: target[0])

all_xray_df = all_xray_df[all_xray_df[condition_labels].sum(axis=1) == 1]

all_xray_df.head()

train_df, test_df = train_test_split(all_xray_df, test_size=0.30, random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)


class XrayDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        address = row['path']
        # Convert image to grayscale
        data = Image.open(address).convert('RGB')
        label = np.array(row['disease_vec'], dtype=np.float32)

        if self.transform:
            data = self.transform(data)

        return data, torch.FloatTensor(label)

    def __len__(self):
        return len(self.data_frame)


class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet121, self).__init__()
        # Load the pre-trained DenseNet121 model
        original_model = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.features = original_model.features
        # Everything else remains the same
        self.classifier = nn.Linear(original_model.classifier.in_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# Initialize the custom model
model = CustomDenseNet121(num_classes=len(condition_labels))

# Move the model to the appropriate device
model = model.to(device=device)

train_transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224 for the model input
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = XrayDataset(train_df, transform=train_transform)
valid_dataset = XrayDataset(valid_df, transform=train_transform)
test_dataset = XrayDataset(test_df, transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

# Data Loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=0,
    shuffle=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    num_workers=0,
    shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    num_workers=0,
    shuffle=False,
)

criterion = nn.CrossEntropyLoss().to(device=device)  # For single-label classification
num_epochs = 20
decay = 1e-4
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=decay)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


def validate(model, valid_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    # val_running_corrects = 0
    val_running_corrects = torch.tensor(0, device=device)  # Initialize as a tensor

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Convert labels to class indices if they are one-hot encoded
            if labels.ndimension() > 1:  # Assuming labels are one-hot encoded
                labels = labels.argmax(dim=1)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Update running loss
            val_running_loss += loss.item()

            # Calculate the accuracy
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data)

    epoch_loss = val_running_loss / len(valid_loader)
    epoch_acc = val_running_corrects.float() / len(valid_loader.dataset)

    return epoch_loss, epoch_acc


def train(epoch):
    model.train()
    running_loss = 0.0
    train_total, train_correct = 0, 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Convert labels to class indices if they are one-hot encoded
        if labels.ndimension() > 1:  # Assuming labels are one-hot encoded
            labels = labels.argmax(dim=1)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        if i % 200 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))

    train_accuracy = 100. * train_correct / train_total
    print(
        f'Epoch {epoch} complete! Average Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%')


for epoch in range(1, num_epochs + 1):
    train(epoch)
    # Implement a validation step here and use its output for the scheduler
    val_loss, val_acc = validate(model, valid_loader, criterion, device)
    scheduler.step(val_loss)  # For ReduceLROnPlateau scheduler


def test(model, data_loader, device):
    model.eval()
    test_predictions = []
    test_labels = []
    test_predictions_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if labels.ndimension() > 1:
                labels = labels.argmax(dim=1)

            # Softmax is applied here by torch.max() as nn.CrossEntropyLoss() expects raw scores
            _, predicted_labels = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)

            test_predictions.append(predicted_labels.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_predictions_probs.append(probs.cpu().numpy())

    test_predictions = np.concatenate(test_predictions)
    test_labels = np.concatenate(test_labels)
    test_predictions_probs = np.concatenate(test_predictions_probs)

    # Here we calculate the overall accuracy
    accuracy = accuracy_score(test_labels, test_predictions)

    # Calculate F1 score using 'weighted' to account for label imbalance
    weighted_f1 = f1_score(test_labels, test_predictions, average='weighted')

    # Print overall test metrics
    print('Test Accuracy: %.4f' % accuracy)
    print('Test F1-score (weighted): %.4f' % weighted_f1)

    # Compute the classification report
    class_report = classification_report(test_labels, test_predictions, target_names=condition_labels, digits=2)

    print('Classification Report:')
    print(class_report)

    return test_labels, test_predictions, test_predictions_probs, accuracy, weighted_f1


test_labels, test_predictions, test_predictions_probs, accuracy, weighted_f1 = test(model, test_loader, device)