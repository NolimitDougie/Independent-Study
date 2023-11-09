import numpy as np
import pandas as pd
import os
import pickle
from glob import glob
import torchvision.models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, EfficientNet_B4_Weights, efficientnet_b4
import seaborn as sns
from tqdm import tqdm
from itertools import cycle
from PIL import Image
from itertools import chain
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import resample
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    multilabel_confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# # Simplifying to 15 primary classes (adding No Finding as the 15th class)
condition_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                    'Effusion', 'Pneumonia', 'Pleural_Thickening',
                    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
for label in condition_labels:
    all_xray_df[label] = all_xray_df['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
all_xray_df.head(20)

all_xray_df['disease_vec'] = all_xray_df.apply(lambda target: [target[condition_labels].values], 1).map(
    lambda target: target[0])

all_xray_df.head()

condition_counts = all_xray_df[condition_labels].sum().sort_values(ascending=False)
minority_labels = condition_counts.tail(5).index.tolist()
minority_df = all_xray_df[all_xray_df[minority_labels].sum(axis=1) == 1]
minority_df = minority_df[minority_df[condition_labels].sum(axis=1) == 1]

balanced_data = []
samples_per_class = 75

for label in condition_labels:
    class_samples = all_xray_df[all_xray_df[label] == 1].sample(samples_per_class, random_state=42)
    balanced_data.append(class_samples)

balanced_df = pd.concat(balanced_data)
balanced_df.reset_index(drop=True, inplace=True)

train_df, test_df = train_test_split(balanced_df, test_size=0.30, random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)


class XrayDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        address = row['path']
        data = Image.open(address).convert('RGB')
        label = np.array(row['disease_vec'], dtype=np.float32)

        if self.transform:
            data = self.transform(data)

        return data, torch.FloatTensor(label)

    def __len__(self):
        return len(self.data_frame)


# Define data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Data Sets
train_dataset = XrayDataset(train_df, transform=train_transform)
valid_dataset = XrayDataset(valid_df, transform=train_transform)
test_dataset = XrayDataset(test_df, transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

minority_train_dataset = XrayDataset(minority_df, transform=train_transform)

minority_train_loader = DataLoader(
    minority_train_dataset,
    batch_size=32,
    num_workers=0,
    shuffle=True,
)


class MultiBranchDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiBranchDenseNet, self).__init__()

        # Loading pre-trained DenseNet121 and removing its classifier
        self.features = models.densenet121(weights=DenseNet121_Weights.DEFAULT).features

        # Main Branch
        self.main_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes),
        )

        # Minority Branch
        self.minority_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, main_x, minority_x=None):
        # Main Branch
        main_x = self.features(main_x)
        main_x = F.adaptive_avg_pool2d(main_x, (1, 1))
        main_x = torch.flatten(main_x, 1)
        main_out = self.main_branch(main_x)

        # If minority_x is None, return only the main_out
        if minority_x is None:
            return main_out, None

        # Minority Branch
        minority_x = self.features(minority_x)
        minority_x = F.adaptive_avg_pool2d(minority_x, (1, 1))
        minority_x = torch.flatten(minority_x, 1)
        minority_out = self.minority_branch(minority_x)

        return main_out, minority_out


model = MultiBranchDenseNet(len(condition_labels))
model = model.to(device)

# Define the loss functions
main_loss_function = nn.BCEWithLogitsLoss().to(device=device)  # For multi-label classification
minority_loss_function = nn.CrossEntropyLoss().to(device=device)  # For single-label classification

num_epochs = 60
decay = 1e-4
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=decay)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


def validate(model, valid_loader):
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0

    with torch.no_grad():
        for main_images, main_labels in valid_loader:
            main_images, main_labels = main_images.to(device), main_labels.to(device)

            # Forward pass
            main_out, _ = model(main_images, main_images)
            loss_main = main_loss_function(main_out, main_labels)

            val_running_loss += loss_main.item()

    return val_running_loss / len(valid_loader)


def find_optimal_thresholds(model, validation_loader, device):
    model.eval()
    # Assuming binary or one-hot encoded labels for multi-label classification
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)

            # Get predictions for both branches, here we assume a single output for simplicity
            main_out, _ = model(images, None)
            predicted_probs = torch.sigmoid(main_out)

            all_labels.append(labels.cpu())
            all_predictions.append(predicted_probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_predictions = torch.cat(all_predictions).numpy()

    optimal_thresholds = []
    for i in range(all_labels.shape[1]):  # Iterate over each class
        precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_predictions[:, i])
        if len(thresholds) > 0:
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            # Find the threshold that maximized the F1 score
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = .10
        optimal_thresholds.append(optimal_threshold)
        print(f"Class {i} optimal threshold: {optimal_threshold}")

    return optimal_thresholds


def train(epoch):
    model.train()
    running_loss = 0.0
    running_main_loss = 0.0
    running_minority_loss = 0.0

    # Cycle the minority_train_loader since it's shorter
    cycled_minority_loader = cycle(minority_train_loader)

    for i, (main_images, main_labels) in enumerate(train_loader):
        # Fetch data from the cycled loader
        minority_images, minority_labels = next(cycled_minority_loader)

        # Send the images and labels to the device (GPU/CPU)
        main_images, main_labels = main_images.to(device), main_labels.to(device)
        minority_images, minority_labels = minority_images.to(device), minority_labels.to(device)

        optimizer.zero_grad()

        # Get outputs from the main branch using main images and from the minority branch using minority images
        main_out, minority_out_from_minority = model(main_images, minority_images)

        # Compute the loss for the main branch
        loss_main = main_loss_function(main_out, main_labels)

        # Compute the loss for the minority branch
        loss_minority = minority_loss_function(minority_out_from_minority, torch.max(minority_labels, 1)[1])

        # Combine the losses with a regularization term for the minority branch
        lambda_reg = 2
        total_loss = loss_main + lambda_reg * loss_minority

        # Backward and optimize
        total_loss.backward()
        optimizer.step()

        # Update the running losses
        running_loss += total_loss.item()
        running_main_loss += loss_main.item()
        running_minority_loss += loss_minority.item()

    # After training steps, get the validation loss
    val_loss = validate(model, valid_loader)
    print(f'Validation Loss: {val_loss:.4f}')

    # Step with the validation loss for the ReduceLROnPlateau scheduler
    scheduler.step(val_loss)

    # Print the final epoch-wise statistics
    print(
        f"Epoch [{epoch}/{num_epochs}], Main Loss: {running_main_loss / len(train_loader):.4f}, "
        f"Minority Loss: {running_minority_loss / len(train_loader):.4f}, "
        f"Total Loss: {running_loss / len(train_loader):.4f}"
    )


# Call the training loop
for epoch in range(1, num_epochs + 1):
    train(epoch)

optimal_thresholds = find_optimal_thresholds(model, valid_loader, device)
print(optimal_thresholds)


def test(model, data_loader, device):
    model.eval()
    test_predictions_list = []
    test_labels_list = []
    test_predictions_probs_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Pass the same batch of images to both branches of your model
            main_out, _ = model(images, None)
            predicted_probs = torch.sigmoid(main_out)
            x = torch.tensor(optimal_thresholds, device=device)
            predicted_labels = (predicted_probs > x).float()

            test_predictions_list.append(predicted_labels.cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())
            test_predictions_probs_list.append(predicted_probs.cpu().numpy())

    test_predictions = np.concatenate(test_predictions_list)
    test_labels = np.concatenate(test_labels_list)

    weighted_f1 = f1_score(test_labels, test_predictions, average='weighted', zero_division=1)

    print('Test F1-score (weighted): %.4f' % weighted_f1)

    class_report = classification_report(test_labels, test_predictions, target_names=condition_labels, digits=2)
    print('Classification Report:')
    print(class_report)

    return test_labels, test_predictions


test_labels, test_predictions = test(model, test_loader, device=device)
