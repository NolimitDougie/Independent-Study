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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix

# Device configuration
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("MPS device found")
else:
    print("MPS device not found.")

# Paths to files and DataEntry file
all_xray_df = pd.read_csv('NihXrayData/Data_Entry_2017.csv')
allImagesGlob = glob('NihXrayData/images*/images/*.png')
bbox_list_df = pd.read_csv('NihXrayData/BBox_List_2017.csv')
bbox_list_df.head(5)
# eof

all_image_paths = {os.path.basename(x): x for x in
                   allImagesGlob}
# print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)

# # Disease Statistics
# num_unique_labels = all_xray_df['Finding Labels'].nunique()
# print('Number of unique labels:', num_unique_labels)
# count_per_unique_label = all_xray_df['Finding Labels'].value_counts()[:15]  # get frequency counts per label
# print(count_per_unique_label)
# exit()

# Data Pre Processing ####
# define condition labels for one hot encoding - simplifying to 14 primary classes (excl. No Finding)
condition_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                    'Effusion', 'Pneumonia', 'Pleural_Thickening',
                    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
for label in condition_labels:
    all_xray_df[label] = all_xray_df['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
all_xray_df.head(20)

all_xray_df['disease_vec'] = all_xray_df.apply(lambda target: [target[condition_labels].values], 1).map(
    lambda target: target[0])
all_xray_df.head()
# eof of one hot encoding

# Data Splitting ###
train_df, test_df = train_test_split(all_xray_df, test_size=0.20, random_state=2020)
#  eof Data Splitting ###


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
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(data), torch.Tensor(label)

    def __len__(self):
        return len(self.data_frame)


# Creating the Dataset and loader
test_dataset = XrayDataset(test_df)
train_dataset = XrayDataset(train_df)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=5000,
    num_workers=0,
    shuffle=True,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=3000,
    num_workers=0,
    shuffle=True,
)

train_dataiter = iter(train_loader)
test_dataiter = iter(test_loader)
train_samples = next(train_dataiter)
test_samples = next(test_dataiter)

train_dataset_3000 = TensorDataset(train_samples[0], train_samples[1])
test_dataset_5000 = TensorDataset(test_samples[0], test_samples[1])

train_loader = DataLoader(train_dataset_3000, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset_5000, batch_size=64, shuffle=False, num_workers=0)

# eof Dataloader #
np.random.seed(42)
torch.manual_seed(42)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Image size 256 * 256 * 3 input channels
        # 1. convolutional layer
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        # 2. convolutional layer
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        # 3. convolutional layer
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        # 4 convolution Layer
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv4_bn = nn.BatchNorm2d(128)
        # output tensor 61 * 61 * 64
        # 5. convolutional layer
        # self.conv5 = nn.Conv2d(64, 128, 3)
        # self.conv5_bn = nn.BatchNorm2d(128)
        # # outputs 59 * 59 * 128 filter images
        # # 6 convolutional layer
        # self.conv6 = nn.Conv2d(128, 128, 3)
        # self.conv6_bn = nn.BatchNorm2d(128)
        # outputs 28 * 28 * 128 filtered Images
        # self.conv7 = nn.Conv2d(128, 128, 3)
        # self.conv7_bn = nn.BatchNorm2d(128)

        # Definition of the MaxPooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # 1. fully-connected layer
        # Input is a flattened 28*28*128 dimensional vector
        self.fc1 = nn.Linear(128 * 61 * 61, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 100)
        self.fc3 = nn.Linear(100, 14)
        # self.output = nn.TransformerDecoder()
        # self.sigmoid = nn.Sigmoid()

        # definition of dropout (dropout probability 25%)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.dropout20(x)
        x = self.conv3_bn(F.relu(self.conv3(x)))
        x = self.pool(self.conv4_bn(F.relu(self.conv4(x))))
        x = self.dropout30(x)
        # x = self.conv5_bn(F.relu(self.conv5(x)))
        # # print(x.shape, " -- after 5th convolution  layer--")
        # x = self.pool(self.conv6_bn(F.relu(self.conv6(x))))
        # x = self.pool(self.conv7_bn(F.relu(self.conv7(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # print(x.shape, "After 1st full connect layer")
        x = self.dropout30(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.sigmoid(self.fc3(x))
        return x


model = ConvNet().to(mps_device)

# Hyper Parameters
num_epochs = 10
weight_decay = 5e-4
learning_rate = 0.001
# eof Hyper Parameters
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train(epoch):
    model.train()
    running_loss = 0.0
    train_total, train_correct = 0.0, 0.0
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(mps_device)
        labels = labels.to(mps_device)

        # Forward pass
        outputs = model(images)
        # outputs = F.sigmoid(outputs)

        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # _, train_predicted = torch.max(outputs.data, 1)
        # _, train_predicted = torch.argmax(y_output)

        train_total += labels.size(0)
        # train_correct += (train_predicted == labels.long()).sum().item()
        # y_train += labels.tolist()
        # y_pred += train_predicted.tolist()

        if i % 200 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))

    # macro_f1 = f1_score(y_train, y_pred, average='macro')
    # print("epoch (%d): Train accuracy: %.4f, f1_score: %.4f, loss: %.3f" % (
    #     epoch, train_correct / train_total, macro_f1, running_loss / train_total))


# Train the model
for epoch in range(1, num_epochs + 1):
    train(epoch)


def test():
    model.eval()
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(mps_device)
            labels = labels.to(mps_device)
            outputs = model(images)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = torch.round(predicted_probs)
            test_predictions.append(predicted_labels.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_predictions = np.concatenate(test_predictions)
    test_labels = np.concatenate(test_labels)

    macro_f1 = f1_score(test_labels, test_predictions, average='macro')
    accuracy = accuracy_score(test_labels, test_predictions)
    print('Test accuracy: %.4f, macro f1_score: %.4f' % (accuracy, macro_f1))

    return test_labels, test_predictions


# Test the model
test_labels, test_predictions = test()

# Confusion Matrix
confusion = multilabel_confusion_matrix(test_labels, test_predictions)
print('Confusion Matrix\n')
print(confusion)
