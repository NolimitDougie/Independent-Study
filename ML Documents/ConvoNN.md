# Dougie Townsell CNN Analysis 

### Architecture of the CNN
```
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

        # defintion of dropout (dropout probability 25%) - to help with overfitting the NN
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
        
```

### Implementation Details 

For the Convolution Neural Network I went with a Convolution filter layer applied followed by Relu activation function and Batch2d layer (batch normalization). 
A MaxPooling layer is applied to scale the image down before going in to the next convolution filter. The dropout layers are there to prevent over fitting with the data 


### HyperParameters
```
num_epochs = 30
batch_size = 4
learning_rate = 0.001
```

## Model Testing 

### Model 1

###  Accuracy & Confusion Matrix
```
[[849   9  26  14   9   1   3   6  53  30]
 [  9 900   0   2   2   1   7   0  18  61]
 [ 47   0 702  34  81  46  59  19   6   6]
 [ 11   2  35 700  50 115  46  21  16   4]
 [  6   0  19  34 864  16  21  36   4   0]
 [  4   2  18 113  29 791  10  30   2   1]
 [  6   0   9  41  23  14 895   4   6   2]
 [ 14   0   7  27  39  28   3 874   3   5]
 [ 35  11   7   4   0   1   5   0 924  13]
 [ 17  27   1   2   0   1   5   4  19 924]]
Accuracy: 0.84
Macro F1-score: 0.84
Micro F1-score: 0.84
```

## Model 2 Testing 

### Model 2

### Architecture of the CNN

```
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1)
        self.conv4 = nn.Conv2d(24, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = (F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv3(x)))  # -> n, 16, 5, 5
        x = (F.relu(self.conv4(x)))
        x = x.view(-1, 32 * 8 * 8)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = self.fc2(x)  # -> n, 10
        return x
```

###  Model Accuracy 

```Accuracy of the network: 71.81 %```











