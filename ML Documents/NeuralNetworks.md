# Machine Learning Neural Networks 

[Types of Neural Networks](https://www.mygreatlearning.com/blog/types-of-neural-networks/
)

[Confusion Matrix](https://www.analyticsvidhya.com/blog/2021/05/in-depth-understanding-of-confusion-matrix/
)

### Image Transformers 
[Explanation on Transformers](https://towardsdatascience.com/using-transformers-for-computer-vision-6f764c5a078b)


## Multilayer Perception 

Simplest Neural Network - composed of an input layer to receive the signal, an output layer that makes a decision or prediction about the input, and in between those two, an arbitrary number of hidden layers that are the true computational engine of the MLP.



## Convolution Neural Network 

This Neural Network is best to use for image classification.

It’s very good at picking up on patterns in the input image, such as lines, gradients, circles, or even eyes and faces. It is this property that makes convolutional neural networks so powerful for computer vision.


### Recurrent Neural Network

[RNN resource](https://towardsdatascience.com/recurrent-neural-networks-explained-with-a-real-life-example-and-python-code-e8403a45f5de)

RNN works on the principle of saving the output of a particular layer and feeding this back to the input in order to predict the output of the layer.

What distinguishes a Recurrent Neural Network from the MultiLayer Perceptron is that a Recurrent Neural Network is built to handle inputs that represent a sequence

Recurrent Neural Networks act like a chain. The computation performed at each time step, depends on the previous computation.

Input state, which captures the input data for the model.

Output state, which captures the results of the model.

Recurrent state, which is in fact a chain of hidden states, and captures all the computations between the input and output states.

RNN mostly used for natural language processing and speech rechonightion 


### Activation Functions 

[Activation Function](https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/)

`ReLU` This means that the neurons will only be deactivated if the output of the linear transformation is less than 0.

`f(x)=max(0,x)` It gives an output x if x is positive and 0 otherwise.

`Sigmoid or Logistic Activation Function` Has an S shaped curve the function exist between 0 and 1 used for models where we have to predict the probability as an output


### Utility Layer i.e Different Types of layers

[Types of layers](https://towardsdatascience.com/four-common-types-of-neural-network-layers-c0d3bb2a966c#:~:text=The%20four%20most%20common%20types,how%20they%20can%20be%20used)

Fully Connected layer 

Convolution Layer 

Deconvolution Layer 

Recurrent Layer  
 
Batch Normalization 

Maxpooling 

Dropout Layer

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

### Stochastic Gradient Descent 

Stochastic Gradient Descent, instead of taking the whole dataset for each iteration, we randomly select batches of data. So your testing a few sample batches from the dataset

First select the initial parameters `w` and the learning rate `n`. Then shuffle the data at each iteration to get the minimum

SDG uses higher number of iterations to reach the local minima thus increasing the overall computation time. 

### Adam Deep learning Optimizer 

Extension of stochastic gradient descent to update network weights during training. Adam optimizer updates the learning rate for each network weight individually

The Algorithm is straight forward to implement, has a faster running time, lower memory requirements, and requires less tuning than any other optimization algorithms 
