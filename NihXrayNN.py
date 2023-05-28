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

# Object Detection ###
A = all_xray_df.set_index('Image Index')
B = bbox_list_df.set_index('Image Index')
list_df = B.join(A, how="inner")
list_df.head(5)
list_df = list_df.reset_index(drop=False)
list_df.head(5)
list_df = list_df.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 11'], axis=1)
list_df.head(5)
# list_df.to_csv('BBox_List.csv', header=True, index=False)
# Writes the data frame to a csv file
print("Object Detections in Chest X-Ray")
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(15, 10), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    img = cv2.imread(list_df.loc[i, 'path'])
    cv2.rectangle(img, (int(list_df.iloc[i, 2:6][0]), int(list_df.iloc[i, 2:6][1])), (
        int(list_df.iloc[i, 2:6][0] + list_df.iloc[i, 2:6][2]), int(list_df.iloc[i, 2:6][1] + list_df.iloc[i, 2:6][3])),
                  (255, 0, 0), 10)
    img = cv2.resize(img, (80, 80))
    ax.imshow(img)
    ax.set_title(list_df.loc[i, 'Finding Label'])
fig.tight_layout()
plt.show()
# eof Object Detection ###

num_unique_labels = all_xray_df['Finding Labels'].nunique()
print('Number of unique labels:', num_unique_labels)

count_per_unique_label = all_xray_df['Finding Labels'].value_counts()[:15]  # get frequency counts per label
print(count_per_unique_label)

# ### Data Pre Processing ####
# define condition labels for one hot encoding - simplifying to 14 primary classes (excl. No Finding)
condition_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                    'Effusion', 'Pneumonia', 'Pleural_Thickening',
                    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']  # taken from paper

for label in condition_labels:
    all_xray_df[label] = all_xray_df['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
all_xray_df.head(20)

all_xray_df['disease_vec'] = all_xray_df.apply(lambda target: [target[condition_labels].values], 1).map(
    lambda target: target[0])

all_xray_df.head()

all_xray_df.to_csv('test.csv', index=False)
# eof of one hot encoding

# Data Splitting ###
train_df, test_df = train_test_split(all_xray_df, test_size=0.20, random_state=2020)


#  eof Data Splitting ###


# Custom X-ray data set for NIH Data
class XrayData(torch.utils.data.Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transforms = transform
        self.len = data_frame.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        address = row['path']
        images = Image.open(address).convert('RGB')
        vec = np.array(row['disease_vec'], dtype=np.float64)  # np.float64 or np.float
        conditions = torch.FloatTensor(vec)
        # if self.transform:
        #     images = self.transform(images)
        return images, conditions


class ToTensor(object):
    def __call__(self, sample):
        labels, img = sample['disease_vec'], sample['path']
        labels = np.array(labels)
        return {'disease_vec': torch.from_numpy(label.long()),
                'path': torch.from_numpy(img).float64()}


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
dsetTrain = XrayData(train_df, transform=transforms)
dsetTest = XrayData(test_df, transform=transforms)
train_loader = torch.utils.data.DataLoader(dataset=dsetTrain, batch_size=64, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(dataset=dsetTest, batch_size=64, shuffle=False, num_workers=1)

print(dsetTest[0])

# show images
np.random.seed(42)
torch.manual_seed(42)


