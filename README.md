### [Machine-Learning](https://www.ibm.com/topics/machine-learning)
___

This code implements a multilayer neural network for classifying irises based on data loaded from the [Scikit-learn library](https://scikit-learn.org/stable/)

> The code does the following:

### Imports required libraries:
* [NumPy](https://numpy.org/install/)
* [TensorFlow](https://www.tensorflow.org/?hl=ru)
* [Matplotlib](https://www.w3schools.com/python/matplotlib_pyplot.asp)
* [Scikit-learn](https://pypi.org/project/scikit-learn/)

___

*   Loads the [Iris](https://www.kaggle.com/code/ash316/ml-from-scratch-with-iris) dataset from Scikit-learn and splits it into train and test sets.
*   Normalizes data for training and testing.
*   Defines a neural network model using the [TensorFlow Keras Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) that contains two fully connected layers: the first layer consists of 10 neurons with a [ReLU activation function](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/), and the second layer consists of 3 neurons with a Softmax activation function
Compiles the model with the [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) optimizer using sparse categorical crossentropy as the loss function and accuracy as the metric.
*  Trains the model on a training set with 50 epochs, using the validation set to evaluate the performance of the model.
*  Saves the trained model to the ['iris_model.h5'](https://www.kaggle.com/code/ash316/ml-from-scratch-with-iris) file.
*  Loads the model from the 'iris_model.h5' file.
*  Creates new prediction data and normalizes it.
*  Makes predictions on new data using the loaded model.
*  Displays prediction results.
*  Generates plots to track model performance, including training and test sample accuracy and loss over epoch.

___

### This script should output predictions (forecasts) for new data in numpy array format like:

> [[9.9983606e-01 1.6389158e-04 1.6122577e-08]
 [1.4681333e-03 8.3092215e-01 1.6760973e-01]
 [2.7367127e-05 3.2064945e-02 9.6790771e-01]]

___
In addition, the script should display graphs to track the performance of the model, including the accuracy and loss on the training and test samples depending on the epoch. Graphs should be displayed using the [plt.show()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html) function.

> Epoch 1/50
4/4 [==============================] - 0s 35ms/step - loss: 1.0664 - accuracy: 0.4000 - val_loss: 0.9859 - val_accuracy: 0.4667
Epoch 2/50
4/4 [==============================] - 0s 5ms/step - loss: 1.0439 - accuracy: 0.4250 - val_loss: 0.9659 - val_accuracy: 0.4667
Epoch 3/50
4/4 [==============================] - 0s 5ms/step - loss: 1.0235 - accuracy: 0.4500 - val_loss: 0.9466 - val_accuracy: 0.5000
Epoch 4/50
4/4 [==============================] - 0s 5ms/step - loss: 1.0047 - accuracy: 0.4750 - val_loss: 0.9279 - val_accuracy: 0.4667
Epoch 5/50
4/4 [==============================] - 0s 5ms/step - loss: 0.9856 - accuracy: 0.4917 - val_loss: 0.9104 - val_accuracy: 0.5000
Epoch 6/50
4/4 [==============================] - 0s 5ms/step - loss: 0.9681 - accuracy: 0.5250 - val_loss: 0.8938 - val_accuracy: 0.5000
Epoch 7/50
4/4 [==============================] - 0s 5ms/step - loss: 0.9502 - accuracy: 0.5250 - val_loss: 0.8784 - val_accuracy: 0.5000
Epoch 8/50
4/4 [==============================] - 0s 5ms/step - loss: 0.9346 - accuracy: 0.5250 - val_loss: 0.8636 - val_accuracy: 0.5000
Epoch 9/50
4/4 [==============================] - 0s 6ms/step - loss: 0.9188 - accuracy: 0.5250 - val_loss: 0.8496 - val_accuracy: 0.4667
Epoch 10/50
4/4 [==============================] - 0s 6ms/step - loss: 0.9048 - accuracy: 0.5333 - val_loss: 0.8362 - val_accuracy: 0.5333
Epoch 11/50
4/4 [==============================] - 0s 6ms/step - loss: 0.8906 - accuracy: 0.5333 - val_loss: 0.8236 - val_accuracy: 0.5333
Epoch 12/50
4/4 [==============================] - 0s 6ms/step - loss: 0.8774 - accuracy: 0.5583 - val_loss: 0.8117 - val_accuracy: 0.5333
Epoch 13/50
4/4 [==============================] - 0s 7ms/step - loss: 0.8647 - accuracy: 0.5833 - val_loss: 0.8002 - val_accuracy: 0.5667
Epoch 14/50
4/4 [==============================] - 0s 5ms/step - loss: 0.8521 - accuracy: 0.5917 - val_loss: 0.7894 - val_accuracy: 0.5667
Epoch 15/50
4/4 [==============================] - 0s 5ms/step - loss: 0.8403 - accuracy: 0.5833 - val_loss: 0.7790 - val_accuracy: 0.5667
Epoch 16/50
4/4 [==============================] - 0s 5ms/step - loss: 0.8290 - accuracy: 0.5917 - val_loss: 0.7690 - val_accuracy: 0.5667
Epoch 17/50
4/4 [==============================] - 0s 5ms/step - loss: 0.8183 - accuracy: 0.6000 - val_loss: 0.7593 - val_accuracy: 0.6000
Epoch 18/50
4/4 [==============================] - 0s 5ms/step - loss: 0.8075 - accuracy: 0.6000 - val_loss: 0.7500 - val_accuracy: 0.6000
Epoch 19/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7969 - accuracy: 0.5917 - val_loss: 0.7411 - val_accuracy: 0.6000
Epoch 20/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7871 - accuracy: 0.5917 - val_loss: 0.7326 - val_accuracy: 0.6000
Epoch 21/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7773 - accuracy: 0.5917 - val_loss: 0.7243 - val_accuracy: 0.6333
Epoch 22/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7679 - accuracy: 0.5917 - val_loss: 0.7162 - val_accuracy: 0.6333
Epoch 23/50
4/4 [==============================] - 0s 6ms/step - loss: 0.7590 - accuracy: 0.5917 - val_loss: 0.7084 - val_accuracy: 0.7000
Epoch 24/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7499 - accuracy: 0.6083 - val_loss: 0.7009 - val_accuracy: 0.7000
Epoch 25/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7413 - accuracy: 0.6167 - val_loss: 0.6935 - val_accuracy: 0.7000
Epoch 26/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7328 - accuracy: 0.6167 - val_loss: 0.6864 - val_accuracy: 0.7000
Epoch 27/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7246 - accuracy: 0.6333 - val_loss: 0.6795 - val_accuracy: 0.7333
Epoch 28/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7168 - accuracy: 0.6333 - val_loss: 0.6729 - val_accuracy: 0.7667
Epoch 29/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7090 - accuracy: 0.6500 - val_loss: 0.6664 - val_accuracy: 0.8000
Epoch 30/50
4/4 [==============================] - 0s 5ms/step - loss: 0.7018 - accuracy: 0.6583 - val_loss: 0.6601 - val_accuracy: 0.8000
Epoch 31/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6942 - accuracy: 0.6583 - val_loss: 0.6540 - val_accuracy: 0.8000
Epoch 32/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6872 - accuracy: 0.6583 - val_loss: 0.6481 - val_accuracy: 0.8000
Epoch 33/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6801 - accuracy: 0.6667 - val_loss: 0.6422 - val_accuracy: 0.8000
Epoch 34/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6733 - accuracy: 0.6750 - val_loss: 0.6366 - val_accuracy: 0.8000
Epoch 35/50
4/4 [==============================] - 0s 6ms/step - loss: 0.6668 - accuracy: 0.6917 - val_loss: 0.6311 - val_accuracy: 0.7667
Epoch 36/50
4/4 [==============================] - 0s 6ms/step - loss: 0.6602 - accuracy: 0.7000 - val_loss: 0.6257 - val_accuracy: 0.7667
Epoch 37/50
4/4 [==============================] - 0s 6ms/step - loss: 0.6538 - accuracy: 0.7000 - val_loss: 0.6204 - val_accuracy: 0.7667
Epoch 38/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6478 - accuracy: 0.7000 - val_loss: 0.6152 - val_accuracy: 0.7667
Epoch 39/50
4/4 [==============================] - 0s 6ms/step - loss: 0.6416 - accuracy: 0.7083 - val_loss: 0.6103 - val_accuracy: 0.7667
Epoch 40/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6356 - accuracy: 0.7250 - val_loss: 0.6055 - val_accuracy: 0.8000
Epoch 41/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6299 - accuracy: 0.7333 - val_loss: 0.6008 - val_accuracy: 0.8000
Epoch 42/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6243 - accuracy: 0.7417 - val_loss: 0.5963 - val_accuracy: 0.8000
Epoch 43/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6188 - accuracy: 0.7417 - val_loss: 0.5918 - val_accuracy: 0.8000
Epoch 44/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6135 - accuracy: 0.7417 - val_loss: 0.5874 - val_accuracy: 0.8000
Epoch 45/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6079 - accuracy: 0.7417 - val_loss: 0.5832 - val_accuracy: 0.8000
Epoch 46/50
4/4 [==============================] - 0s 5ms/step - loss: 0.6027 - accuracy: 0.7500 - val_loss: 0.5790 - val_accuracy: 0.8000
Epoch 47/50
4/4 [==============================] - 0s 5ms/step - loss: 0.5978 - accuracy: 0.7500 - val_loss: 0.5750 - val_accuracy: 0.8000
Epoch 48/50
4/4 [==============================] - 0s 5ms/step - loss: 0.5927 - accuracy: 0.7500 - val_loss: 0.5710 - val_accuracy: 0.8000
Epoch 49/50
4/4 [==============================] - 0s 6ms/step - loss: 0.5878 - accuracy: 0.7500 - val_loss: 0.5672 - val_accuracy: 0.8000
Epoch 50/50