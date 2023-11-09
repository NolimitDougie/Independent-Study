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
    mps_device = torch.device("mps")
else:
    print("MPS device not found.")

# Paths to Images and DataEntry file
all_xray_df = pd.read_csv('NihXrayData/Data_Entry_2017_v2020.csv')
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

train_df, test_df = train_test_split(all_xray_df, test_size=0.30, random_state=2020)


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
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(p=0.3),
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
    batch_size=16,
    num_workers=0,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=0,
    shuffle=False,
)

# # Load pre-trained ResNet50 model
base_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze the parameters of the base model
for param in base_model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one for multi-label classification
num_features = base_model.fc.in_features


# Create custom classifier head
class CustomHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(CustomHead, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Add Dropout layer with 30% drop rate
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.block(x)


# Attach the custom head to the base model
base_model.fc = CustomHead(num_features, hidden_features=256, out_features=15)

# # Create the final model
model = base_model.to(mps_device)

print(model)
exit()

# # Freeze the parameters of the base model
# for param in base_model.parameters():
#     param.requires_grad = False
#
# # Replace the last fully connected layer with a new one for multi-label classification
# num_features = base_model.fc.in_features
# base_model.fc = nn.Linear(num_features, 15)
#
# # Create the final model
# model = base_model.to(mps_device)
# Print the model summary
# print(model)

# Hyperparameters/Loss Function
num_epochs = 1
weight_decay = 1e-4
learning_rate = 0.01

criterion = nn.BCEWithLogitsLoss().to(mps_device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
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


def feedback_on_predictions(y_true, y_pred, label_count):
    if not (0 <= y_true.min() <= y_true.max() <= 1) or not (0 <= y_pred.min() <= y_pred.max() <= 1):
        raise ValueError("Expected binary matrices for y_true and y_pred.")

    # Count correct positive predictions per instance
    correct_positive_count_per_instance = np.sum((y_true == 1) & (y_pred == 1), axis=1)

    # Get the count of each type of correct positive predictions
    count_statistics = Counter(correct_positive_count_per_instance)

    # Display the feedback
    x = list(range(label_count + 1))
    y = []

    for i in x:
        count = count_statistics.get(i, 0)
        percentage = (count / len(y_true)) * 100
        y.append(percentage)
        print(f"{count} X-ray images with exactly {i} correct positive predictions ({percentage:.2f}%).")

    # Plotting
    plt.plot(x, y, marker='o')
    plt.xlabel('Number of True Positive Predictions')
    plt.ylabel('Percentage of Total True Positive Predictions (%)')
    plt.title('Evaluation of True Positive Prediction Correctness')
    plt.xticks(x)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def test(model, data_loader, device):
        model.eval()
        test_predictions = []
        test_labels = []
        class_accuracy = []
        class_f1_score = []
        test_predictions_probs = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predicted_probs = torch.sigmoid(outputs)
                predicted_labels = (predicted_probs > 0.10).float()

                test_predictions.append(predicted_labels.cpu().numpy())
                test_labels.append(labels.cpu().numpy())
                test_predictions_probs.append(predicted_probs.cpu().numpy())

        test_predictions = np.concatenate(test_predictions)
        test_labels = np.concatenate(test_labels)
        macro_f1 = f1_score(test_labels, test_predictions, average='macro', zero_division=1)
        # accuracy = accuracy_score(test_labels, test_predictions)

        # Calculate prediction accuracy, precision, and F1 score for each class
        for i, class_label in enumerate(condition_labels):
            class_accuracy.append(accuracy_score(test_labels[:, i], test_predictions[:, i]))
            class_f1_score.append(f1_score(test_labels[:, i], test_predictions[:, i]))

        print('Model Macro F1-score: %.4f' % macro_f1)
        print('Prediction Metrics per Class:')
        for i, class_label in enumerate(condition_labels):
            print('%s - Accuracy: %.4f, F1-score: %.4f' % (
                class_label, class_accuracy[i], class_f1_score[i]))

        # Compute the classification report
        class_report = classification_report(test_labels, test_predictions, target_names=condition_labels, digits=2)

        print('Classification Report:')
        print(class_report)

        feedback_on_predictions(test_labels, test_predictions, label_count=5)

        return test_labels, test_predictions, class_accuracy, class_f1_score


test_labels, test_predictions, class_accuracy, class_f1_score = test(model, test_loader, mps_device)


def plot_multilabel_confusion_matrices(y_true, y_pred, class_names, cmap=plt.cm.Blues, save_hernia_as_png=False):
    """
    Plot confusion matrices for multilabel data using seaborn's heatmap.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - class_names: Names of the classes/labels
    - cmap: Color map for the heatmap
    - save_hernia_as_png: If True, save the confusion matrix for Hernia as a PNG
    """
    confusion = multilabel_confusion_matrix(y_true, y_pred)

    # Calculate the number of rows and columns for the subplots
    n_labels = len(class_names)
    n_cols = 3  # you can adjust this number based on your preference
    n_rows = ceil(n_labels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))

    # Make sure axes is always a 2D array, even when n_rows is 1
    if n_rows == 1:
        axes = np.reshape(axes, (1, -1))

    for i, cm in enumerate(confusion):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['False', 'True'],
                    yticklabels=['False', 'True'])

        ax.set_title(f"Confusion Matrix: {class_names[i]}")
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        # Check if the current label is 'Hernia' and save it as PNG
        if class_names[i] == "Hernia" and save_hernia_as_png:
            temp_fig = plt.figure()
            temp_ax = temp_fig.add_subplot(111)
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=temp_ax,
                        xticklabels=['False', 'True'],
                        yticklabels=['False', 'True'])
            temp_ax.set_title(f"Confusion Matrix: {class_names[i]}")
            temp_ax.set_ylabel('True label')
            temp_ax.set_xlabel('Predicted label')
            temp_fig.savefig("hernia_confusion_matrix.png")
            plt.close(temp_fig)

    # Delete any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()


# Assuming test_labels and test_predictions are defined
plot_multilabel_confusion_matrices(test_labels, test_predictions, condition_labels, save_hernia_as_png=True)

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
