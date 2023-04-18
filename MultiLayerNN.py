import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score

# fetch_openml grabs datasets by name or dataset id
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
print(X.shape, y.shape)


# Plot the data
def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    n_rows = (len(instances) - 1) // images_per_row + 1
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)
    # np.concatenate merges two or more arrays along a specific axis

    # Reshape the array, so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)

    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap=mpl.cm.binary, **options)
    # imshow displays the data as an image, i.e., on a 2D scale.
    plt.axis("off")


plt.figure(figsize=(4, 4))
example_images = X[:25]
plot_digits(example_images, images_per_row=5)
plt.show()

# Normalize the Data
X = X / 255
y = y.astype(np.uint8)

# Split the data into training set and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


class ConvertDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        sample = {'label': label, 'features': features}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        label, features = sample['label'], sample['features']
        label = np.array(label)
        return {'label': torch.from_numpy(label),
                'features': torch.from_numpy(features).float()}


# Convert training data
train_dataset = ConvertDataset(features=X_train,
                               labels=y_train,
                               transform=transforms.Compose([
                                   ToTensor()
                               ]))

# Load the converted training data into DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

# Convert test data
test_dataset = ConvertDataset(features=X_test,
                              labels=y_test,
                              transform=transforms.Compose([
                                  ToTensor()
                              ]))

# Load the converted training data into DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# dataiter = iter(test_dataloader)
# samples = dataiter.next()
# print(samples['label'], samples['features'])
# print(samples['label'].dtype, samples['features'].dtype)

np.random.seed(42)
torch.manual_seed(42)


class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(784, 300)  # The input layer with 784 features, and the first hidden layer with 300 neurons
        self.fc2 = nn.Linear(300, 100)  # The second hidden layer with 100 neurons
        self.fc3 = nn.Linear(100, 10)  # The output layer with 10 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))  # The first hidden layer uses ReLU activation function
        x = F.relu(self.fc2(x))  # The first hidden layer uses ReLU activation function
        x = self.fc3(x)
        # The output layer does not apply any activation function here, but will be processed directly by loss function
        return x


# HyperParameters
# epochs is the training data iterating through the Neural Network
epochs = 10
learning_rate = 0.01
weight_decay = 5e-4
lossfunction = nn.CrossEntropyLoss()

# Explain this line
# ClassificationNet() is the base class for Neural Networks models should subclass ClassificationNet
model = ClassificationNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
print(model)


# Train the model
#  epochs = 10, so ten iterations of the training data will be passed through
def train(epoch):
    model.train()
    running_loss = 0.0
    train_total, train_correct = 0.0, 0.0
    y_train, y_pred = [], []

    for i, data in enumerate(train_dataloader):
        labels, features = data['label'], data['features']

        optimizer.zero_grad()

        outputs = model(features)

        loss = lossfunction(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, train_predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (train_predicted == labels.long()).sum().item()
        y_train += labels.tolist()
        y_pred += train_predicted.tolist()

    macro_f1 = f1_score(y_train, y_pred, average='macro')
    print("epoch (%d): Train accuracy: %.4f, f1_score: %.4f, loss: %.3f" % (
        epoch, train_correct / train_total, macro_f1, running_loss / train_total))


# Train the model
for epoch in range(1, epochs + 1):
    train(epoch)


# Define the test function
def test():
    model.eval()
    test_correct, test_total = 0.0, 0.0
    y_test, y_pred = [], []

    with torch.no_grad():
        for data in test_dataloader:
            labels, features = data['label'], data['features']

            outputs = model(features)

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

# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

acc = accuracy_score(y_test, y_pred)
macrof1 = f1_score(y_test, y_pred, average='macro')
microf1 = f1_score(y_test, y_pred, average='micro')
print('Accuracy: {:.2f}'.format(acc))
print('Macro F1-score: {:.2f}'.format(macrof1))
print('Micro F1-score: {:.2f}'.format(microf1))
