---
title: planar data classification with one-hidden layer
date: 2018-02-06
copyright: true
categories: English
tags: [neural-networks-deep-learning, deep learning]
mathjax: true
mathjax2: true
---

## Note
These are my personal programming assignments at the third week after studying the course [neural-networks-deep-learning](https://www.coursera.org/learn/neural-networks-deep-learning/) and the copyright belongs to [deeplearning.ai](https://www.deeplearning.ai/).

# planar data classification with one hidden layer
## 1 Packages
Let’s first import all the packages that you will need during this assignment. 
- [numpy](https://blog.csdn.net/koala_tree/article/details/www.numpy.org) is the fundamental package for scientific computing with Python. 
- [sklearn](http://scikit-learn.org/stable/) provides simple and efficient tools for data mining and data analysis. 
- [matplotlib](http://matplotlib.org/) is a library for plotting graphs in Python. 
- testCases_v2 provides some test examples to assess the correctness of your functions 
- planar_utils provide various useful functions used in this assignment


```python
# Package imports
import numpy as np;
import matplotlib.pyplot as plt;
import sklearn;
import sklearn.datasets;
import sklearn.linear_model;
from testCases_v2 import *;
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets;

%matplotlib inline

np.random.seed(1); # set a seed so that the results are consistent
```

You can get the support code from here.

## 2 Dataset
First, let's get the dataset you will work on. The following code will load a “flower” 2-class dataset into variables X and Y.


```python
def load_planar_dataset():  #generate two random array X and Y
    np.random.seed(1)
    m=400     #样本的数量
    N=int(m/2) #每一类的数量，共有俩类数据
    D=2  #维数，二维数据
    X=np.zeros((m,D)) # 生成（m，2）独立的样本
    Y=np.zeros((m,1),dtype='uint8')  #生成（m，1）矩阵的样本
    a=4 #maximum ray of the flower
    for j in range(2):
        ix=range(N*j,N*(j+1))  #范围在[N*j,N*(j+1)]之间
        t=np.linspace(j*3.12,(j+1)*3.12,N)+np.random.randn(N)*0.2  #theta
        r=a*np.sin(4*t)+np.random.randn(N)*0.2  #radius
        X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]   # (m,2),使用np.c_是为了形成（m，2）矩阵
        Y[ix]=j  
    X=X.T   #（2，m）
    Y=Y.T   # (1,m) 
    return X,Y
```

Visualize the dataset using matplotlib. The data looks like a “flower” with some red (label y=0) and some blue (y=1) points. Your goal is to build a model to fit this data.


```python
X,Y = load_planar_dataset();
plt.scatter(X[0,:], X[1,:], c=np.squeeze(Y),s=40,cmap=plt.cm.Spectral);
plt.show();
```


![png](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/output_6_0.png)


You have: 
- a numpy-array (matrix) X that contains your features (x1, x2) 
- a numpy-array (vector) Y that contains your labels (red:0, blue:1).

Lets first get a better sense of what our data is like.

**Exercise**: How many training examples do you have? In addition, what is the shape of the variables X and Y?

**Hint**: How do you get the shape of a numpy array? ([help](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html))


```python
### START CODE HERE ### (≈ 3 lines of code)
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]  # training set size
### END CODE HERE ###

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))
```

    The shape of X is: (2, 400)
    The shape of Y is: (1, 400)
    I have m = 400 training examples!
    

## 3 Simple Logistic Regression
Before building a full neural network, lets first see how logistic regression performs on this problem. You can use sklearn’s built-in functions to do that. Run the code below to train a logistic regression classifier on the dataset.


```python
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, np.squeeze(Y.T));
```

You can now plot the decision boundary of these models. Run the code below.


```python
# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")
```

    Accuracy of logistic regression: 47 % (percentage of correctly labelled datapoints)
    


![png](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/output_12_1.png)


plot_decision_boundary:


```python
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)
```

## 4 Neural Network model
Logistic regression did not work well on the “flower dataset”. You are going to train a Neural Network with a single hidden layer.

Here is our **model**:
![](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/1.png)

**Mathematically**:

For one example $x^{(i)}$: 
$$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1] (i)}\tag{1}$$

$$a^{[1] (i)} = \tanh(z^{[1] (i)})\tag{2}$$

$$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2] (i)}\tag{3}$$

$$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$

{% raw %}
$$y^{(i)}_{prediction} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{5}$$
{% endraw %}

Given the predictions on all the examples, you can also compute the cost $J$ as follows: 

$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$

**Reminder**: 

The general methodology to build a Neural Network is to: 

1. Define the neural network structure ( # of input units, # of hidden units, etc). 
2. Initialize the model’s parameters 
3. Loop: 
    - Implement forward propagation 
    - Compute loss 
    - Implement backward propagation to get the gradients 
    - Update parameters (gradient descent)

You often build helper functions to compute steps 1-3 and then merge them into one function we call `nn_model()`. Once you’ve built `nn_model()` and learnt the right parameters, you can make predictions on new data.

### 4.1 Defining the neural network structure

**Exercise**: Define three variables: 
- $n_x$ : the size of the input layer 
- $n_h$ : the size of the hidden layer (set this to 4) 
- $n_y$ : the size of the output layer

**Hint**: Use shapes of $X$ and $Y$ to find $n_x$ and $n_y$. Also, hard code the hidden layer size to be 4.


```python
# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    n_x = X.shape[0]; # size of input layer
    n_h = 4;
    n_y = Y.shape[0];# size of output layer
    ### END CODE HERE ###
    return (n_x, n_h, n_y);
```


```python
X_assess, Y_assess = layer_sizes_test_case();
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess);
print("The size of the input layer is: n_x = " + str(n_x));
print("The size of the hidden layer is: n_h = " + str(n_h));
print("The size of the output layer is: n_y = " + str(n_y));
```

    The size of the input layer is: n_x = 5
    The size of the hidden layer is: n_h = 4
    The size of the output layer is: n_y = 2
    

### 4.2 Initialize the model’s parameters

**Exercise**: Implement the function `initialize_parameters()`.

**Instructions**: 

- Make sure your parameters' sizes are right. Refer to the neural network figure above if needed. 
- You will initialize the weights matrices with random values. 
- Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b). 
- You will initialize the bias vectors as zeros. 
- Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.


```python
# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.

    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.rand(n_h, n_x) * 0.01;
    b1 = np.zeros((n_h, 1));
    W2 = np.random.rand(n_y, n_h) * 0.01;
    b2 = np.zeros((n_y, 1));
    ### END CODE HERE ###

    assert (W1.shape == (n_h, n_x));
    assert (b1.shape == (n_h, 1));
    assert (W2.shape == (n_y, n_h));
    assert (b2.shape == (n_y, 1));

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2};

    return parameters;
```


```python
n_x, n_h, n_y = initialize_parameters_test_case();
parameters = initialize_parameters(n_x, n_h, n_y);
print("W1 = " + str(parameters["W1"]));
print("b1 = " + str(parameters["b1"]));
print("W2 = " + str(parameters["W2"]));
print("b2 = " + str(parameters["b2"]));
```

    W1 = [[0.00435995 0.00025926]
     [0.00549662 0.00435322]
     [0.00420368 0.00330335]
     [0.00204649 0.00619271]]
    b1 = [[0.]
     [0.]
     [0.]
     [0.]]
    W2 = [[0.00299655 0.00266827 0.00621134 0.00529142]]
    b2 = [[0.]]
    

### 4.3 The Loop

Question: `Implement forward_propagation()`.

**Instructions**: 
- Look above at the mathematical representation of your classifier. 
- You can use the function `sigmoid()`. It is built-in (imported) in the notebook. 
- You can use the function `np.tanh()`. It is part of the numpy library. 
- The steps you have to implement are: 
    1. Retrieve each parameter from the dictionary “parameters” (which is the output of `initialize_parameters()` ) by using `parameters[".."]`. 
    2. Implement Forward Propagation. Compute $Z^{[1]},A^{[1]},Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set). 
- Values needed in the backpropagation are stored in “cache“. The cache will be given as an input to the backpropagation function.


```python
# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"];
    b1 = parameters["b1"];
    W2 = parameters["W2"];
    b2 = parameters["b2"];
    ### END CODE HERE ###

    # Implement Forward Propagation to calculate A2 (probabilities)
    ### START CODE HERE ### (≈ 4 lines of code)
    Z1 = np.dot(W1, X) + b1;
    A1 = np.tanh(Z1);
    Z2 = np.dot(W2, A1) + b2;
    A2 = sigmoid(Z2);
    ### END CODE HERE ###

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2};

    return A2, cache;
```


```python
X_assess, parameters = forward_propagation_test_case();
A2, cache = forward_propagation(X_assess, parameters);
# Note: we use the mean here just to make sure that your output matches ours. 
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']));
```

    0.26281864019752443 0.09199904522700109 -1.3076660128732143 0.21287768171914198
    

**Exercise**: Implement `compute_cost()` to compute the value of the cost $J$.

**Instructions**: 
- There are many ways to implement the **cross-entropy** loss. To help you, we give you how we would have implemented 
- $\sum\limits_{i=0}^{m}  y^{(i)}\log(a^{[2](i)})$


```python
logprobs = np.multiply(np.log(A2),Y);
cost = - np.sum(logprobs); # no need to use a for loop!
```

(you can use either `np.multiply()` and then `np.sum()` or directly `np.dot()`).


```python
# GRADED FUNCTION: compute_cost

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    
    #logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    logprobs = np.dot(np.log(A2), Y.T) + np.dot(np.log(1 - A2), (1 - Y).T);
    cost = - 1.0 / m * logprobs[0][0];
    ### END CODE HERE ###

    cost = np.squeeze(cost);     # makes sure cost is the dimension we expect. 
                                   # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float));

    return cost;
```


```python
A2, Y_assess, parameters = compute_cost_test_case();
print("cost = " + str(compute_cost(A2, Y_assess, parameters)));
```

    cost = 0.6930587610394646
    

Using the cache computed during forward propagation, you can now implement backward propagation.

**Question**: Implement the function `backward_propagation()`.

**Instructions**: 
Backpropagation is usually the hardest (most mathematical) part in deep learning. To help you, here again is the slide from the lecture on backpropagation. You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.

![](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/2.png)

**Tips**: 
To compute $dZ_1$ you'll need to compute $g'^{[1]}(Z^{[1]})$ . 
Since $g^{[1]}(Z^{[1]})$ is the tanh activation function, 
if $a=g^{[1]}(z)$ then $g'^{[1]}(z)=1−a^2$. 
So you can compute $g'^{[1]}(Z^{[1]})$ using `(1 - np.power(A1, 2))`.


```python
# GRADED FUNCTION: backward_propagation

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1];

    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters["W1"];
    W2 = parameters["W2"];
    ### END CODE HERE ###

    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = cache["A1"];
    A2 = cache["A2"];
    ### END CODE HERE ###

    # Backward propagation: calculate dW1, db1, dW2, db2. 
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2 = A2 - Y;
    dW2 = 1 / m * np.dot(dZ2, A1.T);
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True);
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2));
    dW1 = 1 / m * np.dot(dZ1, X.T);
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True);
    ### END CODE HERE ###

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2};

    return grads;
```


```python
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))
```

    dW1 = [[ 0.00301023 -0.00747267]
     [ 0.00257968 -0.00641288]
     [-0.00156892  0.003893  ]
     [-0.00652037  0.01618243]]
    db1 = [[ 0.00176201]
     [ 0.00150995]
     [-0.00091736]
     [-0.00381422]]
    dW2 = [[ 0.00078841  0.01765429 -0.00084166 -0.01022527]]
    db2 = [[-0.16655712]]
    

**Question**: Implement the update rule. Use gradient descent. You have to use (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2).

General gradient descent rule: $θ=θ−α\frac{∂J}{∂θ}$ where $α$ is the learning rate and $θ$ represents a parameter.

**Illustration**: The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.


```python
# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"];
    b1 = parameters["b1"];
    W2 = parameters["W2"];
    b2 = parameters["b2"];
    ### END CODE HERE ###

    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads["dW1"];
    db1 = grads["db1"];
    dW2 = grads["dW2"];
    db2 = grads["db2"];
    ## END CODE HERE ###

    # Update rule for each parameter
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2};

    return parameters;
```


```python
parameters, grads = update_parameters_test_case();
parameters = update_parameters(parameters, grads);

print("W1 = " + str(parameters["W1"]));
print("b1 = " + str(parameters["b1"]));
print("W2 = " + str(parameters["W2"]));
print("b2 = " + str(parameters["b2"]));
```

    W1 = [[-0.00643025  0.01936718]
     [-0.02410458  0.03978052]
     [-0.01653973 -0.02096177]
     [ 0.01046864 -0.05990141]]
    b1 = [[-1.02420756e-06]
     [ 1.27373948e-05]
     [ 8.32996807e-07]
     [-3.20136836e-06]]
    W2 = [[-0.01041081 -0.04463285  0.01758031  0.04747113]]
    b2 = [[0.00010457]]
    

### 4.4 Integrate parts 4.1, 4.2 and 4.3 in nn_model()

**Question**: Build your neural network model in `nn_model()`.

**Instructions**: The neural network model has to use the previous functions in the right order.


```python
# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    ### START CODE HERE ### (≈ 5 lines of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)

        ### END CODE HERE ###

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost));

    return parameters;
```


```python
X_assess, Y_assess = nn_model_test_case();
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True);
print("W1 = " + str(parameters["W1"]));
print("b1 = " + str(parameters["b1"]));
print("W2 = " + str(parameters["W2"]));
print("b2 = " + str(parameters["b2"]));
```

    Cost after iteration 0: 0.693175
    Cost after iteration 1000: 0.000224
    Cost after iteration 2000: 0.000109
    Cost after iteration 3000: 0.000072
    Cost after iteration 4000: 0.000054
    Cost after iteration 5000: 0.000043
    Cost after iteration 6000: 0.000036
    Cost after iteration 7000: 0.000031
    Cost after iteration 8000: 0.000027
    Cost after iteration 9000: 0.000024
    W1 = [[ 0.78668574 -1.44596408]
     [ 0.61841465 -1.14797067]
     [ 0.7941403  -1.45820079]
     [ 0.54249425 -1.01738417]]
    b1 = [[-0.38092208]
     [-0.26640968]
     [-0.38509848]
     [-0.21499134]]
    W2 = [[3.55445121 2.18356796 3.62513709 1.74485812]]
    b2 = [[0.21512924]]
    

### 4.5 Predictions
**Question**: Use your model to predict by building `predict()`. 

Use forward propagation to predict results.

**Reminder**: predictions
$$ y_{prediction} = 
\begin{equation}\begin{cases}
1 & \text{ if activation > 0.5  } \\ 
0 & \text{ otherwise } 
\end{cases}\end{equation}
$$

As an example, if you would like to set the entries of a matrix X to 0 and 1 based on a threshold you would do: `X_new = (X > threshold)`


```python
# GRADED FUNCTION: predict

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    
    A2, cache = forward_propagation(X, parameters);
    predictions = (A2 > 0.5);
    
    ### END CODE HERE ###

    return predictions;
```


```python
parameters, X_assess = predict_test_case();
predictions = predict(parameters, X_assess);
print("predictions mean = " + str(np.mean(predictions)));
```

    predictions mean = 0.6666666666666666
    

It is time to run the model and see how it performs on a planar dataset. Run the following code to test your model with a single hidden layer of `n_h` hidden units.


```python
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True);
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y);
plt.title("Decision Boundary for hidden layer size " + str(4));
```

    Cost after iteration 0: 0.693159
    Cost after iteration 1000: 0.289308
    Cost after iteration 2000: 0.273860
    Cost after iteration 3000: 0.238116
    Cost after iteration 4000: 0.228102
    Cost after iteration 5000: 0.223318
    Cost after iteration 6000: 0.220193
    Cost after iteration 7000: 0.217870
    Cost after iteration 8000: 0.216036
    Cost after iteration 9000: 0.218642
    


![png](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/output_40_1.png)



```python
# Print accuracy
predictions = predict(parameters, X);
print ( ('Accuracy: %d '  %(np.mean(Y == predictions) * 100)) + '%');
```

    Accuracy: 90 %
    

Accuracy is really high compared to Logistic Regression. The model has learnt the leaf patterns of the flower! **Neural networks are able to learn even highly non-linear decision boundaries, unlike logistic regression**.

Now, let’s try out several hidden layer sizes.

### 4.6 Tuning hidden layer size (optional/ungraded exercise)
Run the following code. It may take 1-2 minutes. You will observe different behaviors of the model for various hidden layer sizes.


```python
# This may take about 2 minutes to run

plt.figure(figsize=(16, 32));
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50];
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1);
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000);
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y);
    predictions = predict(parameters, X);
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100);
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy));
```

    Accuracy for 1 hidden units: 67.5 %
    Accuracy for 2 hidden units: 67.25 %
    Accuracy for 3 hidden units: 90.75 %
    Accuracy for 4 hidden units: 90.75 %
    Accuracy for 5 hidden units: 91.25 %
    Accuracy for 20 hidden units: 90.25 %
    Accuracy for 50 hidden units: 91.0 %
    


![png](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/output_44_1.png)


**Interpretation**: 
- The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data. 
- The best hidden layer size seems to be around `n_h = 5`. Indeed, a value around here seems to fits the data well without also incurring noticable overfitting. 
- You will also learn later about regularization, which lets you use very large models (such as n_h = 50) without much overfitting.

**Optional questions**:

Note: Remember to submit the assignment but clicking the blue “Submit Assignment” button at the upper-right.

Some optional/ungraded questions that you can explore if you wish: 
- What happens when you change the tanh activation for a sigmoid activation or a ReLU activation? 
- Play with the learning_rate. What happens? 
- What if we change the dataset? (See part 5 below!)

**You've learnt to:**
- Build a complete neural network with a hidden layer 
- Make a good use of a non-linear unit 
- Implemented forward propagation and backpropagation, and trained a neural network 
- See the impact of varying the hidden layer size, including overfitting.

Nice work!

## 5 Performance on other datasets
If you want, you can rerun the whole notebook (minus the dataset part) for each of the following datasets.


```python
def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3);
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2);
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6);
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None);
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2);

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure;
```


```python
# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles};

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons";
### END CODE HERE ###

X, Y = datasets[dataset];
X, Y = X.T, Y.reshape(1, Y.shape[0]);

# make blobs binary
if dataset == "blobs":
    Y = Y%2;

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral);
```


![png](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/output_47_0.png)



```python

### START CODE HERE ### (choose your dataset)
dataset = "noisy_circles";
### END CODE HERE ###

X, Y = datasets[dataset];
X, Y = X.T, Y.reshape(1, Y.shape[0]);

# make blobs binary
if dataset == "blobs":
    Y = Y%2;

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral);
```


![png](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/output_48_0.png)



```python
### START CODE HERE ### (choose your dataset)
dataset = "blobs";
### END CODE HERE ###

X, Y = datasets[dataset];
X, Y = X.T, Y.reshape(1, Y.shape[0]);

# make blobs binary
if dataset == "blobs":
    Y = Y%2;

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral);
```


![png](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/output_49_0.png)



```python
### START CODE HERE ### (choose your dataset)
dataset = "gaussian_quantiles";
### END CODE HERE ###

X, Y = datasets[dataset];
X, Y = X.T, Y.reshape(1, Y.shape[0]);

# make blobs binary
if dataset == "blobs":
    Y = Y%2;

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral);
```


![png](http://q9kvrafcq.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week3/output_50_0.png)

