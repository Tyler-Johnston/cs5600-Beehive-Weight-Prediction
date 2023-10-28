###########################################
# cs5600_6600_f23_project_1_time_series.py
# some code for project f23
# bugs to vladimir kulyukin on canvas.
###########################################

import numpy as np
import math 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
# from keras.layers.convolutional import Conv1D # this line throws errors
from keras.layers import Conv1D
# from keras.layers.convolutional import MaxPooling1D # this line throws errors
from keras.layers import MaxPooling1D
import matplotlib.pyplot as plt
from aux_funs import partition_dataset_into_samples

def successor(x):
        return x + 1

def plot_preds(plot_title, xlabel, ylabel, y, yhat):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(y, 'r--')
    plt.plot(yhat, 'bs--')
    plt.legend(['y', 'yhat'], loc='best')
    plt.title(plot_title)
    plt.show()

### model 1: simple dense network.
def test_ann_successor():
    # 1. Create a test dataset
    a = [i for i in range(100)]
    b = [i+1 for i in a]
    in_seq  = np.array(a)
    out_seq = np.array(b)

    # 1.a) convert to rows-cols structure
    in_seq  = in_seq.reshape((len(in_seq), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    
    # 1.b) stack columns horizontally
    dataset = np.hstack((in_seq, out_seq))
    # 1.c) choose a number of time steps in a sample
    num_steps = 5

    # 1.c) convert into input/output
    X, y = partition_dataset_into_samples(dataset, num_steps)
    num_features = X.shape[2]
    print('ANN train dataset:')
    print(dataset)

    # 2. Construct ANN model
    model = Sequential()
    model.add(Dense(5, input_shape=(num_steps, num_features), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_features))    
    model.compile(optimizer='adam', loss='mse')

    # 3. fit model
    model.fit(X, y, epochs=2000, verbose=0)

    # 4. save model
    model.save('ann_successor.h5')
    loaded_model = load_model('ann_successor.h5')

    # 5. Construct test dataset
    a2 = [i for i in range(1000, 1021)]
    b2 = [i+1 for i in a2]
    in_seq2  = np.array(a2)
    out_seq2 = np.array(b2)
    in_seq2  = in_seq2.reshape((len(in_seq2), 1))
    out_seq2 = out_seq2.reshape((len(out_seq2), 1))
    dataset2 = np.hstack((in_seq2, out_seq2))
    X2, y2 = partition_dataset_into_samples(dataset2, num_steps)
    print('ANN test dataset:')
    print(dataset2)
    num_features2 = X2.shape[2]

    # 6. test the model on new data
    ground_truth = []
    preds        = []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_steps, num_features))
        y_hat_2 = loaded_model.predict(x_input_2)
        preds.append(y_hat_2[0][0])
        ground_truth.append(y2[i])

    # 7. compute mse and plot
    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    plot_preds('ANN: f(x)=x+1; mse={}'.format(mse), 'x', 'y/yhat', ground_truth, preds)

### model 2: simple conv network.    
def test_convnet_successor():
    # 1. construct a train dataset
    a = [i for i in range(100)]
    b = [i+1 for i in a]
    in_seq  = np.array(a)
    out_seq = np.array(b)

    # 1.a) convert to rows, cols structure
    in_seq  = in_seq.reshape((len(in_seq), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    
    # 1.b) horizontally stack columns
    dataset = np.hstack((in_seq, out_seq))
    print('ConvNet train dataset:')
    print(dataset)
    # 1.c) choose a number of time steps
    num_steps = 5

    # 1.d) convert into input/output
    X, y = partition_dataset_into_samples(dataset, num_steps)
    num_features = X.shape[2]
    
    # 2. Build a ConvNet model
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=2, activation='relu',
                     input_shape=(num_steps, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # 3. fit model
    model.fit(X, y, epochs=2000, verbose=0)

    # 4. save model
    model.save('convnet_successor.h5')
    loaded_model = load_model('convnet_successor.h5')

    # 5. create a test dataset
    a2 = [i for i in range(1000, 1021)]
    b2 = [i+1 for i in a2]
    in_seq2  = np.array(a2)
    out_seq2 = np.array(b2)
    in_seq2  = in_seq2.reshape((len(in_seq2), 1))
    out_seq2 = out_seq2.reshape((len(out_seq2), 1))
    dataset2 = np.hstack((in_seq2, out_seq2))
    print('ConvNet test dataset:')
    print(dataset2)
    X2, y2 = partition_dataset_into_samples(dataset2, num_steps)
    num_features2 = X2.shape[2]

    # 6. test model on new data.
    ground_truth = []
    preds        = []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_steps, num_features))
        y_hat_2 = loaded_model.predict(x_input_2)
        preds.append(y_hat_2[0][0])
        ground_truth.append(y2[i])

    # 7. print stats
    print(ground_truth)
    print(preds)
    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    plot_preds('ConvNet: f(x)=x+1; mse={}'.format(mse), 'x', 'y/yhat', ground_truth, preds)

### LSTM
def test_lstm_successor():
    # 1. construct a train dataset        
    a = [i for i in range(100)]
    b = [i+1 for i in a]
    in_seq  = np.array(a)
    out_seq = np.array(b)

    # 1.a) convert to rows, cols structure
    in_seq  = in_seq.reshape((len(in_seq), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    
    # 1.b) chorizontally stack columns
    dataset = np.hstack((in_seq, out_seq))
    print('LSTM train dataset:')
    print(dataset)
    # 1.c) choose a number of time steps
    num_steps = 5
    # 1.d) convert into input/output
    X, y = partition_dataset_into_samples(dataset, num_steps)
    num_features = X.shape[2]

    # 2. Build a model
    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(num_steps, num_features)))
    model.add(Dense(num_features))
    model.compile(optimizer='adam', loss='mse')
    
    # 3. fit model
    model.fit(X, y, epochs=2000, verbose=0)

    # 4. save model
    model.save('lstm_successor.h5')
    loaded_model = load_model('lstm_successor.h5')

    # 5. Create a train dataset
    a2 = [i for i in range(1000, 1021)]
    b2 = [i+1 for i in a2]
    in_seq2  = np.array(a2)
    out_seq2 = np.array(b2)
    in_seq2  = in_seq2.reshape((len(in_seq2), 1))
    out_seq2 = out_seq2.reshape((len(out_seq2), 1))
    dataset2 = np.hstack((in_seq2, out_seq2))
    print('LSTM train dataset:')
    print(dataset2)
    X2, y2 = partition_dataset_into_samples(dataset2, num_steps)
    num_features2 = X2.shape[2]

    # 6. test model
    ground_truth = []
    preds        = []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_steps, num_features))
        y_hat_2 = loaded_model.predict(x_input_2)
        preds.append(y_hat_2[0][0])
        ground_truth.append(y2[i])

    # 7. compute stats and plot
    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    plot_preds('LSTM: f(x)=x+1; mse={}'.format(mse), 'x', 'y/yhat', ground_truth, preds)

### ================================

def test_ann_2sinx():
    # 1. create training dataset
    a = [math.sin(i) for i in range(100)]
    b = [2*i for i in a]                
    in_seq  = np.array(a)
    out_seq = np.array(b)

    # 1.a) convert to rows-columns structure
    in_seq  = in_seq.reshape((len(in_seq), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    
    # 1.b) horizontally stack columns
    dataset = np.hstack((in_seq, out_seq))
    print('Train dataset for ANN model:')
    print(dataset)
    # 1.c) choose a number of time steps
    num_steps = 5

    # 1.d) convert into input/output
    X, y = partition_dataset_into_samples(dataset, num_steps)
    num_features = X.shape[2]

    # 2. create ANN model
    model = Sequential()
    model.add(Dense(5, input_shape=(num_steps, num_features), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_features))    
    model.compile(optimizer='adam', loss='mse')

    # 3. fit ANN model
    model.fit(X, y, epochs=2000, verbose=0)

    # 4. save trained model
    model.save('ann_2sinx.h5')
    loaded_model = load_model('ann_2sinx.h5')

    # 5. create a distant future dataset
    a2 = [math.sin(i) for i in range(1000, 1021)]
    b2 = [2*i for i in a2]                    
    in_seq2  = np.array(a2)
    out_seq2 = np.array(b2)
    in_seq2  = in_seq2.reshape((len(in_seq2), 1))
    out_seq2 = out_seq2.reshape((len(out_seq2), 1))
    dataset2 = np.hstack((in_seq2, out_seq2))
    print('Test dataset for ANN model:')
    print(dataset2)
    X2, y2 = partition_dataset_into_samples(dataset2, num_steps)
    num_features2 = X2.shape[2]

    # 6. test trained model on the test dataset.
    ground_truth = []
    preds        = []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_steps, num_features))
        y_hat_2 = loaded_model.predict(x_input_2)
        preds.append(y_hat_2[0][0])
        ground_truth.append(y2[i])

    # 7. compute mse and plot
    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    print(mse)
    plot_preds('ANN: sin(x)-->2*sin(x); mse={}'.format(mse), 'x', 'y/yhat', ground_truth, preds)
    
    print('Done...')

def test_convnet_2sinx():
    # 1. create training dataset        
    a = [math.sin(i) for i in range(100)]
    b = [2*i for i in a]                
    in_seq  = np.array(a)
    out_seq = np.array(b)

    # 1.a) convert to rows-columns structure
    in_seq  = in_seq.reshape((len(in_seq), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    
    # 1.b) horizontally stack columns
    dataset = np.hstack((in_seq, out_seq))
    print('Train ConvNet dataset:')
    print(dataset)
    # 1.c) choose a number of time steps
    num_steps = 5

    # 1.d) convert into input/output
    X, y = partition_dataset_into_samples(dataset, num_steps)
    num_features = X.shape[2]

    ## 2. Construct a ConvNet
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=2, activation='relu',
                     input_shape=(num_steps, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # 3. fit model
    model.fit(X, y, epochs=2000, verbose=0)
    
    # 4. save trained model
    model.save('convnet_2sinx.h5')
    loaded_model = load_model('convnet_2sinx.h5')

    # 5. Create a testing dataset.
    a2 = [math.sin(i) for i in range(1000, 1021)]
    b2 = [2*i for i in a2]                    
    in_seq2  = np.array(a2)
    out_seq2 = np.array(b2)
    in_seq2  = in_seq2.reshape((len(in_seq2), 1))
    out_seq2 = out_seq2.reshape((len(out_seq2), 1))
    dataset2 = np.hstack((in_seq2, out_seq2))
    print('Test ConvNet dataset:')
    print(dataset2)
    X2, y2 = partition_dataset_into_samples(dataset2, num_steps)
    num_features2 = X2.shape[2]

    # 6. test trained model on new data
    ground_truth = []
    preds        = []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_steps, num_features))
        y_hat_2 = loaded_model.predict(x_input_2)
        preds.append(y_hat_2[0][0])
        ground_truth.append(y2[i])

    # 7. print mse and plot
    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    print(mse)
    plot_preds('ConvNet: sin(x)-->2*sin(x); mse={}'.format(mse), 'x', 'y/yhat', ground_truth, preds)
    
    print('Done...')    

### lstm
def test_lstm_2sinx():
    # 1. create training dataset
    a = [math.sin(i) for i in range(100)]
    b = [2*i for i in a]                
    in_seq  = np.array(a)
    out_seq = np.array(b)

    # 1.a) convert to rows-columns structure    
    in_seq  = in_seq.reshape((len(in_seq), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    
    # 1.b) horizontally stack columns
    dataset = np.hstack((in_seq, out_seq))
    print('Train LSTM dataset:')
    print(dataset)
    # 1.c) choose a number of time steps
    num_steps = 5

    # 1.d) convert into input/output
    X, y = partition_dataset_into_samples(dataset, num_steps)
    # the dataset knows the number of features, e.g. 2
    num_features = X.shape[2]

    # 2. Construct an LSTM model
    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(num_steps, num_features)))
    model.add(Dense(num_features))
    model.compile(optimizer='adam', loss='mse')

    # 3. fit model
    model.fit(X, y, epochs=2000, verbose=0)
    # 4. save model
    model.save('lstm_2sinx.h5')
    loaded_model = load_model('lstm_2sinx.h5')

    # 5. Create a test dataset
    a2 = [math.sin(i) for i in range(1000, 1021)]
    b2 = [2*i for i in a2]                    
    in_seq2  = np.array(a2)
    out_seq2 = np.array(b2)
    in_seq2  = in_seq2.reshape((len(in_seq2), 1))
    out_seq2 = out_seq2.reshape((len(out_seq2), 1))
    dataset2 = np.hstack((in_seq2, out_seq2))
    print('Test LSTM dataset:')
    print(dataset2)
    X2, y2 = partition_dataset_into_samples(dataset2, num_steps)

    # 6. Test trained model on a new dataset
    ground_truth = []
    preds        = []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_steps, num_features))
        y_hat_2 = loaded_model.predict(x_input_2)
        preds.append(y_hat_2[0][0])
        ground_truth.append(y2[i])

    # 7. print and plot
    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    print(mse)
    plot_preds('LSTM: sin(x)-->2*sin(x); mse={}'.format(mse), 'x', 'y/yhat', ground_truth, preds)
    
    print('Done...')   

if __name__ == '__main__':
   test_ann_successor()     
#    test_convnet_successor()
#    test_lstm_successor()
#    test_ann_2sinx()
#    test_convnet_2sinx()
#    test_lstm_2sinx()
   pass
