import numpy as np
import pandas as pd
import os
import pickle
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torchvision
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from itertools import chain
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    multilabel_confusion_matrix, roc_curve, auc, hamming_loss
from sklearn.utils import resample

# Device configuration
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print("MPS device not found.")

# Paths to Images and DataEntry file
all_xray_df = pd.read_csv('NihXrayData/Data_Entry_2017.csv')
allImagesGlob = glob('NihXrayData/images*/images/*.png')
# eof

all_image_paths = {os.path.basename(x): x for x in
                   allImagesGlob}
# print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)

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
# eof of one hot encoding

# Splitting the Data Frames into 80:20 split ###
train_df, test_df = train_test_split(all_xray_df, test_size=0.30, random_state=2020)
#  eof Data Splitting ###

class_counts2 = train_df[condition_labels].sum()
total_samples = len(train_df)
class_weights = total_samples / (len(condition_labels) * class_counts2)
class_weights_tensor = torch.FloatTensor(class_weights.values)


# Custom X-ray data set for NIH Data
class XrayDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        address = row['path']
        data = Image.open(address).convert('RGB')
        label = np.array(row['disease_vec'], dtype=np.float64)  # np.float64 or np.float

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(data), torch.FloatTensor(label)

    def __len__(self):
        return len(self.data_frame)


# Creating the Dataset for the train & test data frame
test_dataset = XrayDataset(test_df)
train_dataset = XrayDataset(train_df)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=False,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True,
)

# train_dataiter = iter(train_loader)
# test_dataiter = iter(test_loader)
# train_samples = next(train_dataiter)
# test_samples = next(test_dataiter)
#
# train_dataset_3000 = TensorDataset(train_samples[0], train_samples[1])
# test_dataset_5000 = TensorDataset(test_samples[0], test_samples[1])
#
# train_loader = DataLoader(train_dataset_3000, batch_size=64, shuffle=True, num_workers=0)
# test_loader = DataLoader(test_dataset_5000, batch_size=64, shuffle=False, num_workers=0)

# eof Dataloader #
np.random.seed(42)
torch.manual_seed(42)

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#
#         # Image size 256 * 256 * 3 input channels
#         self.conv1 = nn.Conv2d(3, 32, 3)
#         self.conv1_bn = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 32, 3)
#         self.conv2_bn = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 64, 3)
#         self.conv3_bn = nn.BatchNorm2d(64)
#         self.conv4 = nn.Conv2d(64, 128, 3)
#         self.conv4_bn = nn.BatchNorm2d(128)
#
#         self.pool = nn.AvgPool2d(2, 2)
#         # 1. fully-connected layer
#         self.fc1 = nn.Linear(128 * 61 * 61, 128)
#         self.fc1_bn = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 100)
#         self.fc3 = nn.Linear(100, 15)
#
#         # definition of dropout (dropout probability 25%)
#         self.dropout20 = nn.Dropout(0.2)
#         self.dropout30 = nn.Dropout(0.3)
#
#     def forward(self, x):
#         x = self.conv1_bn(F.relu(self.conv1(x)))
#         x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
#         x = self.dropout20(x)
#         x = self.conv3_bn(F.relu(self.conv3(x)))
#         x = self.pool(self.conv4_bn(F.relu(self.conv4(x))))
#         x = self.dropout30(x)
#
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout30(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# model = ConvNet().to(mps_device)

# Set up ResNet 50 model
num_classes = 15  # Number of pathology labels
# Load pre-trained ResNet50 model
base_model = torchvision.models.resnet50(pretrained=True)
# Freeze the parameters of the base model
for param in base_model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one for multi-label classification
num_features = base_model.fc.in_features
base_model.fc = nn.Linear(num_features, 15)

# Create the final model
model = base_model.to(mps_device)
# Print the model summary
print(model)

# Hyper Parameters
num_epochs = 1
weight_decay = 1e-1
learning_rate = 0.001
# eof Hyper Parameters

num_pos_labels = train_df[condition_labels].sum(axis=0)
num_neg_labels = len(train_df) - num_pos_labels
pos_wt = torch.tensor(num_neg_labels / num_pos_labels, dtype=torch.float32)
# Calculate pos_weight for each class
# print(pos_wt, "###### Positive weights ###")

criterion = nn.BCEWithLogitsLoss(weight=class_weights_tensor).to(mps_device)
# criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train(epoch):
    model.train()
    running_loss = 0.0
    train_total, train_correct = 0.0, 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(mps_device)
        labels = labels.to(mps_device)
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        # Loss Function
        loss = criterion(outputs, labels)
        # predicted_labels = (outputs > 0.5).float()
        # Backward and optimize
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # _, train_predicted = torch.argmax(y_output)
        train_total += labels.size(0)
        # train_correct += (train_predicted == labels.long()).sum().item()
        # y_train += labels.tolist()
        # y_pred += train_predicted.tolist()
        # train_correct += (predicted_labels == labels).sum().item()

        if i % 200 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))


# After creating the optimizer, create the learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# Train the model
for epoch in range(1, num_epochs + 1):
    train(epoch)


def test(model, data_loader, device):
    model.eval()
    test_predictions = []
    test_labels = []
    class_accuracy = []
    class_precision = []
    class_f1_score = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.15).float()

            test_predictions.append(predicted_labels.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_predictions = np.concatenate(test_predictions)
    test_labels = np.concatenate(test_labels)
    macro_f1 = f1_score(test_labels, test_predictions, average='macro', zero_division=1)
    accuracy = accuracy_score(test_labels, test_predictions)

    # Calculate prediction accuracy, precision, and F1 score for each class
    for i, class_label in enumerate(condition_labels):
        class_accuracy.append(accuracy_score(test_labels[:, i], test_predictions[:, i]))
        class_precision.append(precision_score(test_labels[:, i], test_predictions[:, i]))
        class_f1_score.append(f1_score(test_labels[:, i], test_predictions[:, i]))

    print('Model Macro F1-score: %.4f' % macro_f1)
    print('Model Accuracy: %.4f' % accuracy)

    print('Prediction Metrics per Class:')
    for i, class_label in enumerate(condition_labels):
        print('%s - Accuracy: %.4f, Precision: %.4f, F1-score: %.4f' % (
            class_label, class_accuracy[i], class_precision[i], class_f1_score[i]))

    # Compute the classification report
    class_report = classification_report(test_labels, test_predictions, target_names=condition_labels, digits=3)

    print('Classification Report:')
    print(class_report)

    return test_labels, test_predictions, class_accuracy, class_precision, class_f1_score


test_labels, test_predictions, class_accuracy, class_precision, class_f1_score = test(model, test_loader, mps_device)

# Multi-Label Confusion Matrix
confusion = multilabel_confusion_matrix(test_labels, test_predictions)
print('Confusion Matrix\n')
for i, cm in enumerate(confusion):
    print(f'Class {condition_labels[i]}:\n{cm}\n')

# create plot
fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
for (i, label) in enumerate(condition_labels):
    fpr, tpr, thresholds = roc_curve(test_labels[:, i].astype(int), test_predictions[:, i])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (label, auc(fpr, tpr)))

# Set labels for plot
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.show()
