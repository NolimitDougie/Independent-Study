import numpy as np
import pandas as pd
import os
import pickle
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
# from keras_preprocessing.image import ImageDataGenerator
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to files and DataEntry file
all_xray_df = pd.read_csv('NihXrayData/Data_Entry_2017.csv')
allImagesGlob = glob('NihXrayData/images*/images/*.png')

all_image_paths = {os.path.basename(x): x for x in
                   allImagesGlob}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)

# # define dummy labels for one hot encoding - simplifying to 14 primary classes (excl. No Finding)
# dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
#                 'Effusion', 'Pneumonia', 'Pleural_Thickening',
#                 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']  # taken from paper
#
# # One Hot Encoding of Finding Labels to dummy_labels
# for label in dummy_labels:
#     all_xray_df[label] = all_xray_df['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
# all_xray_df.head(20)  # check the data, looking good!

num_unique_labels = all_xray_df['Finding Labels'].nunique()
print('Number of unique labels:', num_unique_labels)

count_per_unique_label = all_xray_df['Finding Labels'].value_counts()[:15]  # get frequency counts per label
print(count_per_unique_label)

# ### Data Pre Processing ####
condition_labels = set()


def sep_diseases(x):
    list_diseases = x.split('|')
    for item in list_diseases:
        condition_labels.add(item)
    return list_diseases


# Adds a column to the data frame "disease_vec" and add the condition labels to the data frame
all_xray_df['disease_vec'] = all_xray_df['Finding Labels'].apply(sep_diseases)
condition_labels = list(condition_labels)
condition_labels.sort()

### Condiltion Labels from the dataframe ###

### Data Splitting ###

train_df, test_df = train_test_split(all_xray_df, test_size=0.20, random_state=2020)
print('Number of training examples:', train_df.shape[0])
print('Number of validation examples:', test_df.shape[0])

# # execute just once
train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)

train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')

train_df.head()
print(train_df)


class XrayData(torch.utils.data.Dataset):
    def __init__(self, data_frame, transforms=None):
        self.data_frame = data_frame
        self.transforms = transforms
        self.len = data_frame.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        address = row['path']
        x = Image.open(address).convert('RGB')

        vec = np.array(row['disease_vec'], dtype=float)  # np.float64 or np.float
        y = torch.FloatTensor(vec)

        if self.transforms:
            x = self.transforms(x)
        return x, y


# train_transform = transforms.Compose([
#     transforms.RandomRotation(20),
#     transforms.RandomResizedCrop(224, scale=(0.63, 1)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
# test_transform = transforms.Compose([
#     transforms.Resize(230),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
dsetTrain = XrayData(train_df, transform)
dsetTest = XrayData(test_df, transform)

print(dsetTest)

train_loader = torch.utils.data.DataLoader(dataset=dsetTrain, batch_size=64, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset=dsetTest, batch_size=64, shuffle=False, num_workers=8)


def imshow(img):
    img = img / 2 + 0.48  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# dataiter = iter(train_loader)
# images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
# show images
np.random.seed(42)
torch.manual_seed(42)
