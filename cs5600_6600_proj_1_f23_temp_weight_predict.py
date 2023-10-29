#######################################
# module: temp_weight_predict.py
# bugs to vladimir kulyukin on canvas
#######################################

import numpy as np
import math 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from aux_funs import *

### plotting utility.
def plot_preds(plot_title, xlabel, ylabel, y, yhat):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(y, 'r--')
    plt.plot(yhat, 'bs--')
    plt.legend(['y', 'yhat'], loc='best')
    plt.title(plot_title)
    plt.show()

### this is the function where you should define
### your ANN model for the temp-weight time
### series prediction.
def predict_temp_weight_ann(train_temps, train_weights, test_temps, test_weights,
                            num_steps=5, num_epochs=1000,
                            hiveid=2059,
                            monthid='June',
                            period='P1',
                            saved_model_name='ann_tp_wt.h5'):
    assert len(train_temps) == len(train_weights)

    train_in_seq  = np.array(train_temps)
    train_out_seq = np.array(train_weights)

    train_in_seq  = train_in_seq.reshape((len(train_in_seq), 1))
    train_out_seq = train_out_seq.reshape((len(train_out_seq), 1))
    
    train_dataset = np.hstack((train_in_seq, train_out_seq))

    X, y = partition_dataset_into_samples(train_dataset, num_steps)
    num_features = X.shape[2]

    # ## YOUR ANN MODEL SHOULD BE CONSTRUCTED HERE.
    # model = Sequential()
    # model.add(Dense(10, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(30, activation='relu'))
    # model.add(Dense(30, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='relu'))
    # model.compile(optimizer='adam', loss='mse')

    # Model 1: Basic ANN with ReLU: MSE of .8566 (re-ran got 1.something / weird graph)(output = relu = 1.26ish)
    # model = Sequential()
    # model.add(Dense(8, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # Model 2: Model 2: Basic ANN with tanh: MSE of .99 (graph flat)
    # model = Sequential()
    # model.add(Dense(8, activation='tanh', input_shape=(num_steps, num_features)))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # Model 3: Two-layer ANN with ReLU and tanh: MSE of 1.17 (graph flat)
    # model = Sequential()
    # model.add(Dense(8, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(8, activation='tanh'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # Model 4: Two-layer ANN with tanh and ReLU: MSE of 1.08 (flat)
    # model = Sequential()
    # model.add(Dense(8, activation='tanh', input_shape=(num_steps, num_features)))
    # model.add(Dense(8, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # Model 5: Three-layer ANN with ReLU: MSE of .979
    # model = Sequential()
    # model.add(Dense(8, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # Model 6: Three-layer ANN with tanh: MSE of 1.27 (flat)
    # model = Sequential()
    # model.add(Dense(8, activation='tanh', input_shape=(num_steps, num_features)))
    # model.add(Dense(8, activation='tanh'))
    # model.add(Dense(8, activation='tanh'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # Model 7: Three-layer ANN with Mixed Activation Functions (1.05 mse & flat)
    # model = Sequential()
    # model.add(Dense(8, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(8, activation='tanh'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # Model 8: Four-layer ANN with ReLU mse: 1.33 
    # model = Sequential()
    # model.add(Dense(8, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # Model 9: Four-layer ANN with tanh: 1.25 flat
    # model = Sequential()
    # model.add(Dense(8, activation='tanh', input_shape=(num_steps, num_features)))
    # model.add(Dense(8, activation='tanh'))
    # model.add(Dense(8, activation='tanh'))
    # model.add(Dense(8, activation='tanh'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # mse : .8629 
    # model = Sequential()
    # model.add(Dense(16, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # mse 1.26 relatively flat
    # model = Sequential()
    # model.add(Dense(8, activation='sigmoid', input_shape=(num_steps, num_features)))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # 1.19 flat
    # model = Sequential()
    # model.add(Dense(8, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(8, activation='sigmoid'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # 1.05 flat
    # model = Sequential()
    # model.add(Dense(8, activation='sigmoid', input_shape=(num_steps, num_features)))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(8, activation='tanh'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # model = Sequential()
    # model.add(Dense(8, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(16, activation='sigmoid'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # model = Sequential()
    # model.add(Dense(12, activation='sigmoid', input_shape=(num_steps, num_features)))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # model = Sequential()
    # model.add(Dense(10, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(20, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features))
    # model.compile(optimizer='adam', loss='mse')

    # .98 loss .75
    # model = Sequential()
    # model.add(Dense(16, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # .99 
    # model = Sequential()
    # model.add(Dense(12, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(18, activation='relu'))
    # model.add(Dense(24, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # # .88, 1.21 when re-ran
    # model = Sequential()
    # model.add(Dense(8, activation='relu', input_shape=(num_steps, num_features)))
    # model.add(Dense(12, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_features, activation='linear'))
    # model.compile(optimizer='adam', loss='mse')

    # .95, .97 graph looks clean, loss .75ish. pretty consistent even though it is off-set
    model = Sequential()
    model.add(Dense(4, activation='relu', input_shape=(num_steps, num_features)))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_features, activation='linear'))
    model.compile(optimizer='adam', loss='mse')















    model.fit(X, y, epochs=num_epochs, verbose=1)
    model.save(saved_model_name)
    loaded_model = load_model(saved_model_name)

    test_in_seq  = np.array(test_temps)
    test_out_seq = np.array(test_weights)
    test_in_seq  = test_in_seq.reshape((len(test_in_seq), 1))
    test_out_seq = test_out_seq.reshape((len(test_out_seq), 1))
    dataset2 = np.hstack((test_in_seq, test_out_seq))
    X2, y2 = partition_dataset_into_samples(dataset2, num_steps)
    num_features2 = X2.shape[2]

    ground_truth = []
    preds        = []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_steps, num_features))
        y_hat_2   = loaded_model.predict(x_input_2)
        preds.append(y_hat_2[0][0])
        ground_truth.append(y2[i])

    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    plot_preds('ANN: temp->weight; MSE={}; {},{},{}'.format(mse, hiveid,monthid,period), 'x', 'y/yhat', ground_truth, preds)
    print('Done...')

### this is the function where you should define
### your ConvNet model for the temp-weight time
### series prediction.    
def predict_temp_weight_convnet(train_temps, train_weights, test_temps, test_weights,
                                hiveid=2059,
                                monthid='June',
                                period='P1',                                
                                num_steps=5, num_epochs=1000,
                                saved_model_name='convnet_tp_wt.h5'):
    assert len(train_temps) == len(train_weights)

    print('num_steps == {}'.format(num_steps))
    print('num_epochs == {}'.format(num_epochs))

    train_in_seq  = np.array(train_temps)
    train_out_seq = np.array(train_weights)

    train_in_seq  = train_in_seq.reshape((len(train_in_seq), 1))
    train_out_seq = train_out_seq.reshape((len(train_out_seq), 1))
    
    train_dataset = np.hstack((train_in_seq, train_out_seq))

    X, y = partition_dataset_into_samples(train_dataset, num_steps)
    num_features = X.shape[2]

    ### YOUR CONVNET MODEL SHOULD BE CONSTRUCTED HERE.   
    # 2. Build a ConvNet model
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=2, activation='relu',
                     input_shape=(num_steps, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # # ConvNet Model
    # model = Sequential()
    # # Add 1D Convolutional layer with 64 filters, kernel size of 2 and 'relu' activation function
    # model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(num_steps, num_features)))
    # # Add MaxPooling layer
    # model.add(MaxPooling1D(pool_size=2))
    # # Flatten the output
    # model.add(Flatten())
    # # Add a dense layer with 50 units
    # model.add(Dense(50, activation='relu'))
    # # Add a dense layer with 1 unit (for output)
    # model.add(Dense(1))
    # # Compile the model
    # model.compile(optimizer='adam', loss='mse')    

    model.fit(X, y, epochs=num_epochs, verbose=1)
    model.save(saved_model_name)
    loaded_model = load_model(saved_model_name)

    test_in_seq  = np.array(test_temps)
    test_out_seq = np.array(test_weights)
    test_in_seq  = test_in_seq.reshape((len(test_in_seq), 1))
    test_out_seq = test_out_seq.reshape((len(test_out_seq), 1))
    dataset2 = np.hstack((test_in_seq, test_out_seq))
    X2, y2 = partition_dataset_into_samples(dataset2, num_steps)
    num_features2 = X2.shape[2]

    ground_truth = []
    preds        = []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_steps, num_features))
        y_hat_2   = loaded_model.predict(x_input_2)
        preds.append(y_hat_2[0][0])
        ground_truth.append(y2[i])

    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    plot_preds('ConvNet: temp->weight; MSE={}; {},{},{}'.format(mse,hiveid,monthid,period),
               'x', 'y/yhat', ground_truth, preds)
    print('Done...')


### this is the function where you should define
### your LSTM model for the temp-weight time
### series prediction.        
def predict_temp_weight_lstm(train_temps, train_weights, test_temps, test_weights,
                             hiveid=2059,
                             monthid='June',
                             period='P1',                                                             
                             num_steps=5, num_epochs=1000,
                             saved_model_name='lstm_tp_wt.h5'):
    assert len(train_temps) == len(train_weights)

    train_in_seq  = np.array(train_temps)
    train_out_seq = np.array(train_weights)

    train_in_seq  = train_in_seq.reshape((len(train_in_seq), 1))
    train_out_seq = train_out_seq.reshape((len(train_out_seq), 1))
    
    train_dataset = np.hstack((train_in_seq, train_out_seq))

    X, y = partition_dataset_into_samples(train_dataset, num_steps)
    num_features = X.shape[2]

    ### YOUR LSTM MODEL SHOULD BE DEFINED HERE.
    # 2. Build a model
    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(num_steps, num_features)))
    model.add(Dense(num_features))
    model.compile(optimizer='adam', loss='mse')

    # # LSTM Model
    # model = Sequential()
    # # Add LSTM layer with 50 units and 'relu' activation function. Input shape is (n_steps, n_features)
    # model.add(LSTM(50, activation='relu', input_shape=(num_steps, num_features)))
    # # Add a dense layer with 1 unit (for output)
    # model.add(Dense(num_features))
    # # Compile the model
    # model.compile(optimizer='adam', loss='mse')
    
    model.fit(X, y, epochs=num_epochs, verbose=1)
    model.save(saved_model_name)
    loaded_model = load_model(saved_model_name)

    test_in_seq  = np.array(test_temps)
    test_out_seq = np.array(test_weights)
    test_in_seq  = test_in_seq.reshape((len(test_in_seq), 1))
    test_out_seq = test_out_seq.reshape((len(test_out_seq), 1))
    dataset2 = np.hstack((test_in_seq, test_out_seq))
    X2, y2 = partition_dataset_into_samples(dataset2, num_steps)
    num_features2 = X2.shape[2]

    ground_truth = []
    preds        = []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_steps, num_features))
        y_hat_2   = loaded_model.predict(x_input_2)
        preds.append(y_hat_2[0][0])
        ground_truth.append(y2[i])

    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    plot_preds('LSTM: temp->weight; MSE={}; {},{},{}'.format(mse,hiveid,monthid,period),
               'x', 'y/yhat', ground_truth, preds)    
    print('Done...')        

## split_data takes the first 70 percent of the data and returns
## the training temperature series, the testing temperature series,
## the training weight series, the testing weight series.
## the pre_process can be set to cubic_root or log10 in aux_funs.py.
def split_data(hiveid, monthid, period='P3', train_percent=0.7,
               pre_process=None):
    fp = 'periods_p1_p2_p3_p4_p5/{}/BeePi2022_TP_WT_{}_{}_{}.csv'.format(hiveid,
                                                                         hiveid,
                                                                         monthid,
                                                                         period)
    recs = csv_file_to_arys(fp, TW_INDEX)
    temps, weights = get_tw_recs(recs)
    if pre_process is not None:
       temps = [pre_process(t) for t in temps]
       weights = [pre_process(w) for w in weights]
    split_inx = math.floor(len(temps) * train_percent)
    train_temps, test_temps = temps[:split_inx], temps[split_inx:]
    train_weights, test_weights = weights[:split_inx], weights[split_inx:]
    return train_temps, test_temps, train_weights, test_weights

### these functions call predict_weight_temp_ann,
### predict_weight_temp_convnet, and predict_weight_temp_lstm with
### the appropriate values.
def train_weight_temp_ann(hiveid=2059, monthid='June', period='P3', num_steps=12,
                          pre_process=None, num_epochs=2000,
                          model_name='ann_tp_to_wt.h5'):
    print('train_weight_temp_ann...')
    train_temps, test_temps, train_weights, test_weights = split_data(hiveid,
                                                                      monthid,
                                                                      period=period,
                                                                      pre_process=pre_process)
    predict_temp_weight_ann(train_temps, train_weights, test_temps, test_weights,
                            num_steps=num_steps,
                            num_epochs=num_epochs,
                            monthid=monthid,
                            period=period,
                            hiveid=hiveid,
                            saved_model_name=model_name)

def train_weight_temp_convnet(hiveid=2059, monthid='June', period='P3', num_steps=12,
                              pre_process=None, num_epochs=2000,
                              model_name='convnet_tp_to_wt.h5'):
    train_temps, test_temps, train_weights, test_weights = split_data(hiveid,
                                                                      monthid,
                                                                      period=period,
                                                                      pre_process=pre_process)
    predict_temp_weight_convnet(train_temps, train_weights, test_temps, test_weights,
                                hiveid=hiveid,
                                monthid=monthid,
                                period=period,
                                num_steps=num_steps,
                                num_epochs=num_epochs,
                                saved_model_name=model_name)

def train_weight_temp_lstm(hiveid=2059, monthid='June', period='P3', num_steps=12,
                              pre_process=None, num_epochs=2000,
                              model_name='lstm_tp_to_wt.h5'):
    train_temps, test_temps, train_weights, test_weights = split_data(hiveid,
                                                                      monthid,
                                                                      period=period,
                                                                      pre_process=pre_process)
    predict_temp_weight_lstm(train_temps, train_weights, test_temps, test_weights,
                             hiveid=hiveid,
                             monthid=monthid,
                             period=period,                             
                             num_steps=num_steps,
                             num_epochs=num_epochs,
                             saved_model_name=model_name)

'''
monthid='June', 'July', 'August', 'September'
period='P1', 'P2', 'P3', 'P4', 'P5'
hiveid=2059

DateTime 2120	2137	2059   2123	2141	2142	2129	2158	2130	2146    
period 1 -- 6:00  -- 11:55
period 2 -- 12:00 -- 15:55
period 3 -- 16:00 -- 19:55
period 4 -- 20:00 -- 23:55
period 5 -- 00:00 -- 05:55
hiveid =  [2120, 2137, 2059, 2123, 2141, 2142, 2129, 2158, 2130, 2146]
monthid = ['June', 'July', 'August', 'September']
'''

if __name__ == '__main__':

   ### uncomment to run.
   
   train_weight_temp_ann(hiveid=2059, monthid='June', period='P3',
                         num_steps=12,
                         pre_process=None,
                         num_epochs=2000,
                         model_name='ann_tp_to_wt.h5')

#    train_weight_temp_ann(hiveid=2120, monthid='July', period='P4',
#                          num_steps=12,
#                          pre_process=None,
#                          num_epochs=2000,
#                          model_name='ann_tp_to_wt.h5')

#    train_weight_temp_convnet(hiveid=2059, monthid='June', period='P3',
#                              num_steps=12,
#                              pre_process=None,
#                              num_epochs=2000,
#                              model_name='convnet_tp_to_wt.h5')   
   
#    train_weight_temp_lstm(hiveid=2059, monthid='June', period='P3',
#                           num_steps=12,
#                           pre_process=None,
#                           num_epochs=2000,
#                           model_name='lstm_tp_to_wt.h5')

   pass

