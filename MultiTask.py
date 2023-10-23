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
from torchvision.models import ResNet50_Weights
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
all_xray_df = pd.read_csv('NihXrayData/Data_Entry_2017_v2020.csv')
allImagesGlob = glob('NihXrayData/images*/images/*.png')
# eof

all_image_paths = {os.path.basename(x): x for x in
                   allImagesGlob}
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

disease_counts = all_xray_df[condition_labels].sum().sort_values(ascending=False)

# Filter out rows that have only one label
single_label_df = all_xray_df[all_xray_df[condition_labels].sum(axis=1) == 1]

# Create separate dataframes for each label
label_dfs = {label: single_label_df[single_label_df[label] == 1] for label in condition_labels}

# Sort the labels based on counts
sorted_labels = sorted(disease_counts.index, key=lambda x: disease_counts[x], reverse=True)

# Divide the sorted labels into batches
batch1_labels = sorted_labels[:5]
batch2_labels = sorted_labels[5:10]
batch3_labels = sorted_labels[10:]

# Group dataframes for each batch
batch1_data = pd.concat([label_dfs[label] for label in batch1_labels])
batch2_data = pd.concat([label_dfs[label] for label in batch2_labels])
batch3_data = pd.concat([label_dfs[label] for label in batch3_labels])
# For multi-label instances
multi_label_data = all_xray_df[all_xray_df[condition_labels].sum(axis=1) > 1]

print(multi_label_data[condition_labels].sum())