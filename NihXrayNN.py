import numpy as np
import pandas as pd
import os
import pickle
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torchvision
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to files and DataEntry file
all_xray_df = pd.read_csv('NihXrayData/Data_Entry_2017.csv')
allImagesGlob = glob('NihXrayData/images*/images/*.png')
bbox_list_df = pd.read_csv('NihXrayData/BBox_List_2017.csv')
bbox_list_df.head(5)

all_image_paths = {os.path.basename(x): x for x in
                   allImagesGlob}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)

# # Object Detection ###
# A = all_xray_df.set_index('Image Index')
# B = bbox_list_df.set_index('Image Index')
# list_df = B.join(A, how="inner")
# list_df.head(5)
# list_df = list_df.reset_index(drop=False)
# list_df.head(5)
# list_df = list_df.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 11'], axis=1)
# list_df.head(5)
# # list_df.to_csv('BBox_List.csv', header=True, index=False)
# # Writes the data frame to a csv file
# print("Object Detections in Chest X-Ray")
# fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(15, 10), subplot_kw={'xticks': [], 'yticks': []})
# for i, ax in enumerate(axes.flat):
#     img = cv2.imread(list_df.loc[i, 'path'])
#     cv2.rectangle(img, (int(list_df.iloc[i, 2:6][0]), int(list_df.iloc[i, 2:6][1])), (
#         int(list_df.iloc[i, 2:6][0] + list_df.iloc[i, 2:6][2]), int(list_df.iloc[i, 2:6][1] + list_df.iloc[i, 2:6][3])),
#                   (255, 0, 0), 10)
#     img = cv2.resize(img, (80, 80))
#     ax.imshow(img)
#     ax.set_title(list_df.loc[i, 'Finding Label'])
# fig.tight_layout()
# plt.show()
# eof Object Detection ###
# Disease Statistics
# num_unique_labels = all_xray_df['Finding Labels'].nunique()
# print('Number of unique labels:', num_unique_labels)
# count_per_unique_label = all_xray_df['Finding Labels'].value_counts()[:15]  # get frequency counts per label
# print(count_per_unique_label)

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
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(data), torch.tensor(label)

    def __len__(self):
        return len(self.data_frame)


# Creating the Dataset and loader
test_dataset = XrayDataset(test_df)
train_dataset = XrayDataset(train_df)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True,
)
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
        # outputs - (32 filter images, 254 * 254)
        # 2. convolutional layer
        # sees 254 * 254 * 32 tensor (2x2 MaxPooling layer beforehand)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        # outputs 126 * 126 * 32 filtered images, kernel-size is 3
        # 3. convolutional layer
        # sees 126 x 126 x 32 tensor (2x2 MaxPooling layer beforehand)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        # outputs 124 * 124 * 64 filtered images, kernel-size is 3
        # 4 Convolution Layer
        # 124 * 124 * 64 Image
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        # output tensor 61 * 61 * 64
        # 5. convolutional layer
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        # outputs 59 * 59 * 128 filter images
        # 6 convolutional layer
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(128)
        # outputs 28 * 28 * 128 filtered Images

        # Definition of the MaxPooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # 1. fully-connected layer
        # Input is a flattened 4*4*64 dimensional vector
        # Output is 500 dimensional vector
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

        # definition of dropout (dropout probability 25%)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1_bn(F.relu(self.conv1(x)))
        print(x.shape, "After first layer")
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        print(x.shape, "After second layer")
        x = self.dropout20(x)
        x = self.conv3_bn(F.relu(self.conv3(x)))
        print(x.shape, "After third layer")
        x = self.pool(self.conv4_bn(F.relu(self.conv4(x))))
        print(x.shape, "After fourth layer")
        x = self.dropout30(x)
        x = self.conv5_bn(F.relu(self.conv5(x)))
        print(x.shape, "After fifth layer")
        x = self.pool(self.conv6_bn(F.relu(self.conv6(x))))
        print(x.shape, "After 6 layer")
        x = self.dropout40(x)

        # flatten output of third convolutional layer into a vector
        # this vector is passed through the fully-connected nn
        print(x.shape, "After 6 layer and dropout layer")
        x = x.view(-1, 128 * 28 * 28)
        # add dropout layer
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout50(x)
        # add 2nd hidden layer, without relu activation function
        x = self.fc2(x)
        return x


model = ConvNet().to(device)

num_epochs = 10
weight_decay = 5e-4
learning_rate = 0.001

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train(epoch):
    model.train()
    running_loss = 0.0
    train_total, train_correct = 0.0, 0.0
    y_train, y_pred = [], []
    loss_hist, acc_hist = [], []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, train_predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (train_predicted == labels.long()).sum().item()
        y_train += labels.tolist()
        y_pred += train_predicted.tolist()

        if i % 2000 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))

    macro_f1 = f1_score(y_train, y_pred, average='macro')
    print("epoch (%d): Train accuracy: %.4f, f1_score: %.4f, loss: %.3f" % (
        epoch, train_correct / train_total, macro_f1, running_loss / train_total))


# Train the model
for epoch in range(1, num_epochs + 1):
    train(epoch)


def test():
    model.eval()
    test_correct, test_total = 0.0, 0.0
    y_test, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels.long()).sum().item()
            y_test += labels.tolist()
            y_pred += predicted.tolist()

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print('Test accuracy: %.4f, macro f1_score: %.4f' % (test_correct / test_total, macro_f1))

    return y_test, y_pred


# Test the model
y_test, y_pred = test()

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

acc = accuracy_score(y_test, y_pred)
macrof1 = f1_score(y_test, y_pred, average='macro')
microf1 = f1_score(y_test, y_pred, average='micro')
print('Accuracy: {:.2f}'.format(acc))
print('Macro F1-score: {:.2f}'.format(macrof1))
print('Micro F1-score: {:.2f}'.format(microf1))
