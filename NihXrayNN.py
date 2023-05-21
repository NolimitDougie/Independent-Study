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
from keras_preprocessing.image import ImageDataGenerator


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

# # # once saved , use the following statements to load train and test dataframes subsequently
train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')

train_df.head()

print(train_df)

# IMG_SIZE = (128, 128)
# all_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
#
# core_idg = ImageDataGenerator(rescale=1.0/255.0, validation_split = 0.06)
#
# # obtain the training images using the above generator
# train_gen = core_idg.flow_from_dataframe(
#         dataframe=train_df,
#         directory=None,
#         x_col='path',
#         y_col=all_labels,
#         target_size=(128, 128),
#         batch_size=64,
#         class_mode='raw',
#         classes=all_labels,
#         shuffle=True,
#         color_mode = "grayscale",
#         subset='training')
#
# # obtain the validation images using the above generator
# test_gen = core_idg.flow_from_dataframe(
#         dataframe=test_df,
#         directory=None,
#         x_col='path',
#         y_col=all_labels,
#         target_size=(128, 128),
#         batch_size=64,
#         class_mode='raw',
#         classes=all_labels,
#         shuffle=False,
#         color_mode = "grayscale",
#         subset='validation')

