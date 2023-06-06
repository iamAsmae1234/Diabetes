# Diabetes
# Introduction:
Thanks to the evolution of technology like Artificial Intelligence (AI), Cloud, Big Data, Blockchain and Internet of Things, our living world is continuously and rapidly changing. 
In this day, these technologies have invaded in a very significant way our daily life in almost all areas.
AI and Big Data are two booming technologies with a lot of application promise for businesses in all industries.
These two technologies are strongly linked, indeed big data is generally the starting point of any AI strategy.
AI is a scientific discipline that seeks through a set of computer programs to create and simulate human intelligence, such as understanding natural languages, reasoning, visual or auditory perception, etc. 
AI is finding its place in more and more areas of our daily lives: smarter cars, safer road traffic, more accurate weather forecasts, better medical and computational diagnostics, etc.

# 1 General background
  # 1.1 Objective :
    Predict/explain the occurrence of diabetes (variable to be predicted) based on the characteristics of people (age, BMI, etc.) (explanatory variables).
  # The context:
    This dataset comes from the National Institute of Diabetes and Digestive and Kidney Diseases. 
    The purpose of the dataset is to diagnostically predict whether a patient has diabetes or not, based on certain diagnostic measures included in the dataset. 
    Several constraints were imposed on the selection of these instances from a larger database. 
    In particular, all of the patients here are women of at least 21 years of age of Pima Indian origin.
  # Content:
    The datasets consist of several medical predictor variables and one target variable, the outcome. 
    Predictor variables include the number of pregnancies the patient has had, her BMI, insulin level, age, etc.
    
# 2 Part 1:

  As a first approach to solving the regression problem, we provide you with the architecture to follow:
    1- Prepare your deals and clean them if necessary.
    2- First of all, we will split our data into a training set (X_train, y_train) and a test set (X_test, y_test).

    3- We use, we will use all kinds of regressions, logistic regression, Ridge regression or SVM, KNN.

    Other supervised approaches are available in scikit-learn.
    The goal is to find the best model for this problem.

    4- Use CROSS VALIDATION to test your models.

    5- Performance measures by confrontation between Y and Y^: confusion matrix + measures
   
# 2 Part 2:
  # 2.2.1 Keras:
    Keras is one of the most powerful and easy-to-use Python libraries for deep learning models and which allows the use of neural networks in a simple way. 
    Keras encompasses the Theano and TensorFlow numerical computation libraries. 
    To define a deep learning model, we define the following characteristics:
      • Number of layers;
      • Types of layers;
      • Number of neurons in each layer;
      • Activation functions of each layer;
      • Size of inputs and outputs.
      The type of layers that we will use are the dense type.
      Dense layers are the most popular because it is an ordinary neural network layer where each of its neurons is connected to the neurons of the previous layer and the next layer.
      The most used functions are the Rectified Linear Unit (ReLU) function, the Sigmoid function and the linear function.
      As for the number of neurons in the input layer, it is the same as the number of features in the dataset. In the rest of this project, we will try to build a deep learning model.

          1- To apply a deep learning model, the dataset must only contain numeric variables.

          • First, the data type of each column of the dataset is displayed.
          2- Construction of the deep learning model

Now that the data is preprocessed, we'll start building our model.
We create an instance of Sequential() from the Keras library, this overlays a set of layers to create a single model. We pass a list of layers that we want to use for our model as a parameter.
As you will notice in this model, we have created several dense layers and a single dropout type layer. The first layer is the input layer, its number of neurons is equal to the number of features in the dataset.
In each layer there are 64 neurons, this number is the optimal result of several tests. Indeed, 64 neurons per layer for the example of this dataset gives a fairly accurate result.
# Noticed :
It is recommended to try several numbers until you get accurate results.
For the Dropout layer, the number of input data has been reduced by 30% in order to avoid the phenomenon of overfitting. The seed takes a value of 2 to have more reproducible results.
Finally, the last dense layer with a single neuron is the output layer. By default, it takes the linear activation function.
    modele = keras.Sequential([layers.Dense(64, activation = 'relu', input_shape = [train.shape[1]]),
    ...
# Part 3:
  4.1 Graphics and ergonomics:

Export the best model and link it with a GUI of your choice.
As a suggestion you can use Qt .

PyQt is a free module that links the Python language with the Qt library distributed under two licenses: a commercial one and the GNU GPL. 
It allows to create graphical interfaces in Python. 
An extension of Qt Creator (graphical utility for creating Qt interfaces) makes it possible to generate Python code for graphical interfaces.




