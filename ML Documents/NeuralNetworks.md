# Machine Learning Neural Networks 

[Types of Neural Networks](https://www.mygreatlearning.com/blog/types-of-neural-networks/
)

## Multilayer Perception 

Simplest Neural Network - composed of an input layer to receive the signal, an output layer that makes a decision or prediction about the input, and in between those two, an arbitrary number of hidden layers that are the true computational engine of the MLP.


## Deep vs Wide Networks 







## Convolution Neural Network 

This Neural Network is best to use for image classification.

It’s very good at picking up on patterns in the input image, such as lines, gradients, circles, or even eyes and faces. It is this property that makes convolutional neural networks so powerful for computer vision.


### Activation Functions 

[Activation Function](https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/)

`ReLU` This means that the neurons will only be deactivated if the output of the linear transformation is less than 0.

`f(x)=max(0,x)` It gives an output x if x is positive and 0 otherwise.

### Utility Layer i.e Different Types of layers

Fully Connected layer 

Convolution Layer

Deconvolution Layer 

Recurrent Layer 


## Optimizing Techniques

`Epoch` - The number of times the algorithm runs on the whole training dataset

`Batch` - Denotes the number of samples to be taken for updating model parameters. i.e a batch size of 64 images means your passing 64 images through the neural network before your starting your back propaganda 

`Learning rate` - It is a parameter that provides the model a scale of how much weights should be updated

`Cost Function/Loss Function` - calculates the difference between the predicted value and the actual value


### Exhaustive search

Exhaustive search, or brute-force search, is the process of looking for the most optimal hyperparameters by checking whether each candidate is a good match.

### Gradient descent

Gradient descent is the most common algorithm for model optimization for minimizing the error. In order to perform gradient descent, you have to iterate over the training dataset while re-adjusting the model.

`Mini-batch` gradient descent is a variation of the gradient descent algorithm that splits the training dataset into small batches that are used to calculate model error and update model coefficients