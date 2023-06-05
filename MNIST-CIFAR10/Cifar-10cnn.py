import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Device configuration
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print("MPS device not found.")

# Hyper-parameters
num_epochs = 50
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=32)])

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=train_transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

print(test_loader)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1. convolutional layer
        # sees 32x32x3 image tensor, i.e 32x32 RGB pixel image
        # outputs 32 filtered images, kernel-size is 3
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        # 2. convolutional layer
        # sees 16x16x32 tensor (2x2 MaxPooling layer beforehand)
        # outputs 32 filtered images, kernel-size is 3
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        # 3. convolutional layer
        # sees 8x8x32 tensor (2x2 MaxPooling layer beforehand)
        # outputs 64 filtered images, kernel-size is 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(128)

        # Definition of the MaxPooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # 1. fully-connected layer
        # Input is a flattened 4*4*64 dimensional vector
        # Output is 500 dimensional vector
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

        # defintion of dropout (dropout probability 25%)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)

    def forward(self, x):
        # Pass data through a sequence of 3 convolutional layers
        # Firstly, filters are applied -> increases the depth
        # Secondly, Relu activation function is applied
        # Finally, MaxPooling layer decreases width and height
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.dropout20(x)
        x = self.conv3_bn(F.relu(self.conv3(x)))
        x = self.pool(self.conv4_bn(F.relu(self.conv4(x))))
        x = self.dropout30(x)
        x = self.conv5_bn(F.relu(self.conv5(x)))
        x = self.pool(self.conv6_bn(F.relu(self.conv6(x))))
        x = self.dropout40(x)

        # flatten output of third convolutional layer into a vector
        # this vector is passed through the fully-connected nn
        x = x.view(-1, 128 * 4 * 4)
        # add dropout layer
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout50(x)
        # add 2nd hidden layer, without relu activation function
        x = self.fc2(x)
        return x


model = ConvNet()

model.to(mps_device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)


def train(epoch):
    model.train()
    running_loss = 0.0
    train_total, train_correct = 0.0, 0.0
    y_train, y_pred = [], []
    loss_hist, acc_hist = [], []
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(mps_device)
        labels = labels.to(mps_device)

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
            print('Epoch: loss: {:.6f}'.format(
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
            images = images.to(mps_device)
            labels = labels.to(mps_device)
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
