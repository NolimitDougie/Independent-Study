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

# Device configuration GPU support for MAC
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print("MPS device not found.")

# Paths to Images and DataEntry file
# all_xray_df = pd.read_csv('NihXrayData/Data_Entry_2017.csv')
all_xray_df = pd.read_csv('NihXrayData/Data_Entry_2017_v2020.csv')
allImagesGlob = glob('NihXrayData/images*/images/*.png')
# eof

all_image_paths = {os.path.basename(x): x for x in
                   allImagesGlob}
# print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)

# # # Filter out rows with 'No Finding'
# all_xray_df = all_xray_df[all_xray_df['Finding Labels'] != 'No Finding']
# # Reset the index
# all_xray_df.reset_index(drop=True, inplace=True)

# # Data Pre Processing ####
# # Simplifying to 15 primary classes (adding No Finding as the 15th class)
condition_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                    'Effusion', 'Pneumonia', 'Pleural_Thickening',
                    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']
for label in condition_labels:
    all_xray_df[label] = all_xray_df['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
all_xray_df.head(20)

all_xray_df['disease_vec'] = all_xray_df.apply(lambda target: [target[condition_labels].values], 1).map(
    lambda target: target[0])

all_xray_df.head()

balanced_data = []
# Target Samples for each class
samples_per_class = 220

for label in condition_labels:
    class_samples = all_xray_df[all_xray_df[label] == 1].sample(samples_per_class, random_state=42)
    balanced_data.append(class_samples)

# Concatenate the balanced data samples for all classes
balanced_df = pd.concat(balanced_data)

# Reset the index of the new DataFrame
balanced_df.reset_index(drop=True, inplace=True)
# all_xray_df = all_xray_df.sample(10000)


# 70:30 split for Training and Testing data
train_df, test_df = train_test_split(balanced_df, test_size=0.30, random_state=2020)


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
    # transforms.RandomResizedCrop(224),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(p=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Data Sets
train_dataset = XrayDataset(train_df, transform=train_transform)
test_dataset = XrayDataset(test_df, transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]))

# Data Loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=False,
)

# # Load pre-trained ResNet50 model
# base_model = torchvision.models.resnet50(pretrained=True)
# # Freeze the parameters of the base model
# for param in base_model.parameters():
#     param.requires_grad = False
#
# # Replace the last fully connected layer with a new one for multi-label classification
# num_features = base_model.fc.in_features
# base_model.fc = nn.Linear(num_features, 15)
#

# Create the final model
model = base_model.to(mps_device)
# Print the model summary
# print(model)

# Hyperparameters/Loss Function
num_epochs = 10
weight_decay = 1e-4
learning_rate = 0.0001
# learning_rate = 0.01

criterion = nn.BCEWithLogitsLoss().to(mps_device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# eof Hyper Parameters/Loss Function
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train(epoch):
    model.train()
    running_loss = 0.0
    train_total, train_correct = 0.0, 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(mps_device)
        labels = labels.to(mps_device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        # Backward and optimize

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_total += labels.size(0)

        if i % 200 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))
        # Step the scheduler after each epoch
    scheduler.step()


for epoch in range(1, num_epochs + 1):
    train(epoch)


def test(model, data_loader, device):
    model.eval()
    test_predictions = []
    test_labels = []
    class_accuracy = []
    class_f1_score = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.17).float()

            test_predictions.append(predicted_labels.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_predictions = np.concatenate(test_predictions)
    test_labels = np.concatenate(test_labels)
    macro_f1 = f1_score(test_labels, test_predictions, average='weighted', zero_division=1)
    accuracy = accuracy_score(test_labels, test_predictions)

    # Calculate prediction accuracy, precision, and F1 score for each class
    for i, class_label in enumerate(condition_labels):
        class_accuracy.append(accuracy_score(test_labels[:, i], test_predictions[:, i]))
        class_f1_score.append(f1_score(test_labels[:, i], test_predictions[:, i]))

    print('Model Macro F1-score: %.4f' % macro_f1)
    print('Model Accuracy: %.4f' % accuracy)

    print('Prediction Metrics per Class:')
    for i, class_label in enumerate(condition_labels):
        print('%s - Accuracy: %.4f, F1-score: %.4f' % (
            class_label, class_accuracy[i], class_f1_score[i]))

    # Compute the classification report
    class_report = classification_report(test_labels, test_predictions, target_names=condition_labels, digits=3)

    print('Classification Report:')
    print(class_report)

    return test_labels, test_predictions, class_accuracy, class_f1_score


test_labels, test_predictions, class_accuracy, class_f1_score = test(model, test_loader, mps_device)

# Multi-Label Confusion Matrix
confusion = multilabel_confusion_matrix(test_labels, test_predictions)
print('Confusion Matrix\n')
for i, cm in enumerate(confusion):
    print(f'Class {condition_labels[i]}:\n{cm}\n')

# Create plot
fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))

# Calculate ROC curve and AUC for each class
for i, label in enumerate(condition_labels):
    fpr, tpr, thresholds = roc_curve(test_labels[:, i], test_predictions[:, i])
    auc_score = roc_auc_score(test_labels[:, i], test_predictions[:, i])

    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (label, auc_score))

# Set labels for plot
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.show()
