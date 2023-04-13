## [Machine-Learning](https://www.ibm.com/topics/machine-learning)

This code implements a multilayer neural network for classifying irises based on data loaded from the Scikit-learn library.
[link](https://scikit-learn.org/stable/)

___

> The code does the following:

### Imports required libraries:

* [NumPy](https://numpy.org/install/),
* [TensorFlow](https://www.tensorflow.org/?hl=ru),
* [Matplotlib](https://www.w3schools.com/python/matplotlib_pyplot.asp),
* [Scikit-learn](https://pypi.org/project/scikit-learn/).

___

* Loads the [Iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) dataset from Scikit-learn and splits it into train and test sets.
* Normalizes data for training and testing.
* Defines a neural network model using the [TensorFlow Keras Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) that contains two fully connected layers: the first layer consists of 10 neurons with a ReLU activation function, and the second layer consists of 3 neurons with a Softmax activation function.
* Compiles the model with the Adam optimizer using sparse categorical cross entropy as the loss function and accuracy as the metric.
* Trains the model on a training set with 50 epochs, using the validation set to evaluate the performance of the model.
* Saves the trained model to the 'iris_model.h5' file.
* Loads the model from the 'iris_model.h5' file.
* Creates new prediction data and normalizes it.
* Makes predictions on new data using the loaded model.
* Display prediction results.
* Generates plots to track model performance, including training and test sample accuracy and loss over epoch.
