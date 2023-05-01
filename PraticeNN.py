import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Convert the dataset into Tensor used by PyTorch

transform = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 64

# Download the MINST data directly from PyTorch
# The downloaded datasets are stored in the same folder with this jupyter notebook file
# For train dataset, use "train=True"
# For test dataset, use "train=False"
train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

# Load the datasets into DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# One batch has 64 images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))

np.random.seed(42)
torch.manual_seed(42)


# nn.model is the base class for all neural network modules
# nn.linear applies linear transformation to the incoming data
class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)  # 3 input channel to 10 channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 10 channels to 20 channels
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    # Relu Activation function gives an output x if x is positive and 0 otherwise.
    # Max_pool
    def forward(self, x):
        x = F.relu(self.conv1(x))  # Use ReLU as activation function
        x = F.max_pool2d(x, 2)  # Apply max_pooling on the output of the convolution layer
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Hyperparamters
# epochs is # of iterations of training data running through the neural network
epochs = 3
learning_rate = 0.01
weight_decay = 5e-4
# Decreases the learning rate as the network see's an image more than once
lossfunction = nn.CrossEntropyLoss()

model = ClassificationNet()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print(model)


def train(epoch):
    model.train()

    running_loss = 0.0
    train_total, train_correct = 0.0, 0.0
    y_train, y_pred = [], []

    for i, (features, labels) in enumerate(train_dataloader):

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

        if i % 200 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, i * len(features), len(train_dataloader.dataset),
                       100. * i / len(train_dataloader), loss.item()))

    macro_f1 = f1_score(y_train, y_pred, average='macro')
    print("epoch (%d): Train accuracy: %.4f, f1_score: %.4f, loss: %.3f" % (
        epoch, train_correct / train_total, macro_f1, running_loss / train_total))


# Train the model
for epoch in range(1, epochs + 1):
    train(epoch)


def test():
    model.eval()

    test_correct, test_total = 0.0, 0.0
    y_test, y_pred = [], []

    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = model(features)
            # max returns (value ,index)
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
