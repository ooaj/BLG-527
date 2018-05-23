import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model

# import data as pandas data frame, change according to your path
start = time.time()
train_data = pd.read_csv("/home/ooa/Belgeler/git/BLG-527/hw3/Data/optdigits.tra", header=None)
test_data = pd.read_csv("/home/ooa/Belgeler/git/BLG-527/hw3/Data/optdigits.tes", header=None)

train_data = train_data.values
test_data = test_data.values

y_train = train_data[:,64]
y_test = test_data[:,64]

X_train = train_data[:,:64]
X_test = test_data[:,:64]

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

X_train = X_train.reshape((3823, 8, 8))
X_train = X_train.reshape((3823, 8, 8, 1))
X_train = X_train.astype('float32') / 16

X_test = X_test.reshape((1797, 8, 8))
X_test = X_test.reshape((1797, 8, 8, 1))
X_test = X_test.astype('float32') / 16

'''X_val = X_val.reshape((765, 8, 8))
X_val = X_val.reshape((765, 8, 8, 1))
X_val = X_val.astype('float32') / 16
'''

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#y_val= to_categorical(y_val)

def dnn(optimizer='rmsprop'):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(8, 8, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn = dnn, verbose=0)

batch_size = [32, 64, 128]
epochs = [5, 10, 20]
optimizers = ['rmsprop', 'adam', 'Adagrad']
param_grid = dict(optimizer=optimizers, batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)

cvres = grid_result.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(8, 8, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.Dense(10, activation='softmax'))
model2.compile(optimizer=grid_result.best_params_['optimizer'],loss='categorical_crossentropy',
                metrics=['accuracy'])

history2 = model2.fit(X_train, y_train, validation_split=0.2, epochs=grid_result.best_params_['epochs'],
                      batch_size=grid_result.best_params_['batch_size'])

acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

y_pred_test_2 = model2.predict_classes(X_test)
y_pred_train_2 = model2.predict_classes(X_train)

test_loss, test_acc = model2.evaluate(X_test, y_test)
print("Test accuracy is ", test_acc)

train_loss, train_acc = model2.evaluate(X_train, y_train)
print("Train accuracy is ", train_acc)

end = time.time()
print(end - start)

weight_conv2d_1 = model2.layers[0].get_weights()[0][:,:,0,:]

'''http://www.codeastar.com/visualize-convolutional-neural-network/
'''
col_size = 6
row_size = 5
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(12,8))
for row in range(0,row_size): 
    for col in range(0,col_size):
        ax[row][col].imshow(weight_conv2d_1[:,:,filter_index],cmap="gray")
        filter_index += 1

deneme_7 = test_data[1, :64]
deneme_7 = np.reshape(deneme_7, (8,8))
plt.imshow(deneme_7, interpolation='Nearest')
plt.show()

from keras.models import Model
layer_outputs = [layer.output for layer in model2.layers]
activation_model = Model(inputs=model2.input, outputs=layer_outputs)
activations = activation_model.predict(X_test)
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size): 
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

# Model Visualization
plot_model(model2, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Confusion Matrices
train_confusion = confusion_matrix(train_data[:,64], y_pred_train_2)
valid_confusion = confusion_matrix(test_data[:,64], y_pred_test_2)

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)