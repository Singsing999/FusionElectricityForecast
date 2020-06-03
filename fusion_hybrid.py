# multichannel multi-step cnn for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from numpy import NaN
from numpy import savetxt
from numpy import arange
from numpy import mean
from numpy import append
from numpy import corrcoef
from numpy import reshape
from numpy import repeat
from pandas import read_csv
from pandas import set_option
from pandas import concat
from pandas import DataFrame
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import pyplot
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.ensemble import RandomForestRegressor
#import plaidml.keras
#plaidml.keras.install_backend()

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input, Flatten, merge, Reshape, Dropout, Masking
from keras.layers.convolutional import Conv1D
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import MaxPooling1D
from keras.utils import plot_model
from sklearn.model_selection import KFold # import KFold
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

if os.name != 'nt':
 config = tf.ConfigProto()
 config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
 config.log_device_placement = True  # to log device placement (on which device the operation runs)
 sess = tf.Session(config=config)
 set_session(sess)  # set this TensorFlow session as the default session for Keras

#from cnn_univar import repeat_evaluate

# split a univariate dataset into train/test sets
def split_dataset(train, test, n_input, n_output):
  # split into standard weeks
  #n_train_days = 3500 #1000
  #train, test = data[0:n_train_days], data[n_train_days:4060]#1100]
  # restructure into windows of weekly data
  #train = array(split(train, len(train)/n_output))
  test = array(split(test, len(test)/n_output))
  return test
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
  scores_mse = list()
  scores_rmse = list()
  scores_mae = list()
  scores_r2 = list()
  # calculate an RMSE score for each day
  for i in range(predicted.shape[1]): 
      # calculate mse
      if(actual.ndim != predicted.ndim):
        actual = actual.reshape(actual.shape[0], actual.shape[1]*actual.shape[2])
      try:
        mse = mean_squared_error(actual[:, i], array([abs(ele) for ele in predicted[:, i]]))
        mae = mean_absolute_error(actual[:, i], array([abs(ele) for ele in predicted[:, i]]))
        r2Score = r2_score(actual[:, i], array([abs(ele) for ele in predicted[:, i]]))
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores_rmse.append(rmse)
        scores_mse.append(mse)
        scores_mae.append(mae)
        scores_r2.append(r2Score)
      except:
        mse = 0
        mae = 0
        rmse = 0
        r2Score = 0
 
  return scores_mse, scores_rmse, scores_mae, scores_r2

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out, ftrIndex):
  # flatten data
  data = train#.reshape((train.shape[0]*train.shape[1], train.shape[2]))
  X, y = list(), list()
  in_start = 0
  # step over the entire history one time step at a time
  for _ in range(len(data)):
    # define the end of the input sequence
    in_end = in_start + n_input
    out_end = in_end + n_out
    # ensure we have enough data for this instance
    if out_end < len(data):
      X.append(data[in_start:in_end, :])
      y.append(data[in_end:out_end, ftrIndex])
    # move along one time step
    in_start += 1
  return array(X), array(y)
# train the model

def build_model_var(train, n_input, n_output, ftrIndex):
  # prepare data
  print("build_model::train "+str(train.shape))
  # define model
  model = VAR(train)
  return model

def build_model_rf(train, n_input, n_output, ftrIndex):
  # prepare data
  train_x, train_y = to_supervised(train, n_input, n_output, ftrIndex)
  train_x = train_x.reshape(train_x.shape[0], train_x.shape[1])
  #train_y = train_y.reshape(1, train_y.shape[0])
  # Train the model on training data
  # define model
  rf = RandomForestRegressor(n_estimators = 5, random_state = 10, criterion='mse', max_depth=15)
  return rf.fit(train_x, train_y)

def build_model_fusion_cnn_lstm(train_d1, n_input, n_output, ftrIndx_m1):
  
  # define parameters
  verbose, epochs, batch_size = 1, 5, 200

  #MODEL 1 CNN-LSTM Takes weather+electric data
  train_x_1, train_y_1 = to_supervised(train_d1, n_input, n_output, ftrIndex_m1)
  n_timesteps, n_features, n_outputs = train_x_1.shape[1], train_x_1.shape[2], train_y_1.shape[1]
  # define model
  model1 = Sequential()

  model1.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
  model1.add(MaxPooling1D(pool_size=2))

  model1.add(LSTM(15, activation='relu',
            #return_sequences=True, 
            input_shape=(n_timesteps, n_features)))
  #model.add(Dropout(0.2))
  model1.add(Dense(n_outputs))

  model1.add(Reshape((n_output,1)))
  ####

  #MODEL 2 - CNN - Takes only electric data
  n_features = 1
  # define model
  model2 = Sequential()
  #print("features: "+str(n_features))
  model2.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
  model2.add(MaxPooling1D(pool_size=2))

  model2.add(Flatten())
  model2.add(Dense(n_outputs))
  model2.add(Reshape((n_output,1)))

  #CONCATENATE
  merged = concatenate([model1.output, model2.output], axis=2)

  #LINEAR REGRESSION
  linReg = Dense(10, input_dim=2, kernel_initializer='normal', activation='relu')(merged)
  linReg = Dense(1, kernel_initializer='normal') (linReg)

  #COMPLETE MODEL
  fusion = Model(inputs=[model1.input, model2.input], outputs=linReg)
  plot_model(fusion, show_shapes=True, show_layer_names=False, to_file='fusion_lstm_cnn_lstm.pdf')
  fusion.compile(loss='mse', optimizer='adam')
  
  train_y_1 = train_y_1.reshape(train_y_1.shape[0], train_y_1.shape[1], 1)
  train_x_2 = train_x_1[0:int(train_x_1.shape[0]*3/4), :, ftrIndex_m1]
  train_x_2 = train_x_2.reshape(train_x_2.shape[0], train_x_2.shape[1], 1)
  train_x_2_val = train_x_1[int(train_x_1.shape[0]*3/4):, :, ftrIndex_m1]
  train_x_2_val = train_x_2_val.reshape(train_x_2_val.shape[0], train_x_2_val.shape[1], 1)

  # fit network
  history = fusion.fit([train_x_1[0:int(train_x_1.shape[0]*3/4)], train_x_2], train_y_1[0:int(train_y_1.shape[0]*3/4)], validation_data=([train_x_1[int(train_x_1.shape[0]*3/4):], train_x_2_val], train_y_1[int(train_y_1.shape[0]*3/4):]), epochs=epochs, batch_size=batch_size, verbose=verbose)
  #ERROR PLOT####
  pyplot.ylabel('Error')
  pyplot.xlabel('Epochs')
  #pyplot.title("CNN")
  pyplot.plot(history.history['loss'], label="Train")
  pyplot.plot(history.history['val_loss'], label="Validation")
  pyplot.legend()
  #pyplot.show()
  pyplot.savefig('Fusion_CNNLSTM_Train_error.pdf')
  pyplot.close()
  return fusion

def build_model_fusion_lstm_lstm(train_d1, n_input, n_output, ftrIndx_m1):
  
  # define parameters
  verbose, epochs, batch_size = 1, 5, 200

  #MODEL 1 - LSTM_CNN Takes weather+electric data
  train_x_1, train_y_1 = to_supervised(train_d1, n_input, n_output, ftrIndex_m1)
  n_timesteps, n_features, n_outputs = train_x_1.shape[1], train_x_1.shape[2], train_y_1.shape[1]

  # define model
  model1 = Sequential()

  # Encoder LSTM
  model1.add(LSTM(5, activation='relu',
            #return_sequences=True, 
            input_shape=(n_timesteps, n_features)
            ))
  #model.add(Dropout(0.2))
  #model1.add(Dense(n_outputs))

  # Decoder LSTM
  model1.add(RepeatVector(n_outputs))
  model1.add(LSTM(5, activation='relu',
        return_sequences=True))
  #model.add(Dropout(0.2))
  model1.add(TimeDistributed(Dense(n_outputs, activation='relu')))
  model1.add(Dense(1))
  model1.add(Reshape((n_output,1)))
  ####

  #MODEL 2 - CNN - Takes only electric data
  n_features = 1
  # define model
  model2 = Sequential()
  #print("features: "+str(n_features))
  model2.add(Conv1D(filters=16, kernel_size=9, activation='relu', input_shape=(n_timesteps,n_features)))
  model2.add(MaxPooling1D(pool_size=4))

  model2.add(Flatten())
  model2.add(Dense(n_outputs))
  model2.add(Reshape((n_output,1)))

  #CONCATENATE
  merged = concatenate([model1.output, model2.output], axis=2)

  #LINEAR REGRESSION
  linReg = Dense(10, input_dim=2, kernel_initializer='normal', activation='relu')(merged)
  linReg = Dense(1, kernel_initializer='normal') (linReg)

  #COMPLETE MODEL
  fusion = Model(inputs=[model1.input, model2.input], outputs=linReg)
  plot_model(fusion, show_shapes=True, show_layer_names=False, to_file='fusion_lstm_cnn.pdf')
  fusion.compile(loss='mse', optimizer='adam')
  
  train_y_1 = train_y_1.reshape(train_y_1.shape[0], train_y_1.shape[1], 1)
  train_x_2 = train_x_1[0:int(train_x_1.shape[0]*3/4), :, ftrIndex_m1]
  train_x_2 = train_x_2.reshape(train_x_2.shape[0], train_x_2.shape[1], 1)
  train_x_2_val = train_x_1[int(train_x_1.shape[0]*3/4):, :, ftrIndex_m1]
  train_x_2_val = train_x_2_val.reshape(train_x_2_val.shape[0], train_x_2_val.shape[1], 1)

  # fit network
  history = fusion.fit([train_x_1[0:int(train_x_1.shape[0]*3/4)], train_x_2], train_y_1[0:int(train_y_1.shape[0]*3/4)], validation_data=([train_x_1[int(train_x_1.shape[0]*3/4):], train_x_2_val], train_y_1[int(train_y_1.shape[0]*3/4):]), epochs=epochs, batch_size=batch_size, verbose=verbose)
  #ERROR PLOT####
  pyplot.ylabel('Error')
  pyplot.xlabel('Epochs')
  #pyplot.title("CNN")
  pyplot.plot(history.history['loss'], label="Train")
  pyplot.plot(history.history['val_loss'], label="Validation")
  pyplot.legend()
  #pyplot.show()
  pyplot.savefig('Fusion_LSTMLSTM_Train_error.pdf')
  pyplot.close()
  return fusion

def build_model_fusion_cnn_lstm_cnn(train_d1, n_input, n_output, ftrIndx_m1):
  
  # define parameters
  verbose, epochs, batch_size = 1, 5, 200

  #MODEL 1 - CNN_LSTM_CNN Takes weather+electric data
  train_x_1, train_y_1 = to_supervised(train_d1, n_input, n_output, ftrIndex_m1)
  n_timesteps, n_features, n_outputs = train_x_1.shape[1], train_x_1.shape[2], train_y_1.shape[1]
  
  # define model
  model1 = Sequential()
  model1.add(Conv1D(filters=16, kernel_size=9, activation='relu', input_shape=(n_timesteps,n_features)))
  model1.add(MaxPooling1D(pool_size=4))

  model1.add(Flatten())
  model1.add(RepeatVector(n_outputs))
  # LSTM
  model1.add(LSTM(5, activation='relu',
        return_sequences=True))
  model1.add(Dense(n_outputs))
  ####

  model1.add(Conv1D(filters=16, kernel_size=9, activation='relu', input_shape=(n_timesteps,n_features)))
  model1.add(MaxPooling1D(pool_size=4))

  model1.add(Flatten())
  model1.add(Dense(n_outputs))
  model1.add(Reshape((n_output,1)))

  #MODEL 2 - CNN - Takes only electric data
  n_features = 1
  # define model
  model2 = Sequential()
  #print("features: "+str(n_features))
  model2.add(Conv1D(filters=16, kernel_size=9, activation='relu', input_shape=(n_timesteps,n_features)))
  model2.add(MaxPooling1D(pool_size=4))
  model2.add(Flatten())

  model2.add(Dense(n_outputs))
  model2.add(Reshape((n_output,1)))

  #CONCATENATE
  merged = concatenate([model1.output, model2.output], axis=2)

  #LINEAR REGRESSION
  linReg = Dense(10, input_dim=2, kernel_initializer='normal', activation='relu')(merged)
  linReg = Dense(1, kernel_initializer='normal') (linReg)

  #COMPLETE MODEL
  fusion = Model(inputs=[model1.input, model2.input], outputs=linReg)
  plot_model(fusion, show_shapes=True, show_layer_names=False, to_file='fusion_cnn_lstm_cnn.pdf')
  fusion.compile(loss='mse', optimizer='adam')
  train_y_1 = train_y_1.reshape(train_y_1.shape[0], train_y_1.shape[1], 1)
  train_x_2 = train_x_1[0:int(train_x_1.shape[0]*3/4), :, ftrIndex_m1]
  train_x_2 = train_x_2.reshape(train_x_2.shape[0], train_x_2.shape[1], 1)
  train_x_2_val = train_x_1[int(train_x_1.shape[0]*3/4):, :, ftrIndex_m1]
  train_x_2_val = train_x_2_val.reshape(train_x_2_val.shape[0], train_x_2_val.shape[1], 1)

  # fit network
  history = fusion.fit([train_x_1[0:int(train_x_1.shape[0]*3/4)], train_x_2], train_y_1[0:int(train_y_1.shape[0]*3/4)], validation_data=([train_x_1[int(train_x_1.shape[0]*3/4):], train_x_2_val], train_y_1[int(train_y_1.shape[0]*3/4):]), epochs=epochs, batch_size=batch_size, verbose=verbose)
  #ERROR PLOT####
  pyplot.ylabel('Error')
  pyplot.xlabel('Epochs')
  #pyplot.title("CNN")
  pyplot.plot(history.history['loss'], label="Train")
  pyplot.plot(history.history['val_loss'], label="Validation")
  pyplot.legend()
  #pyplot.show()
  pyplot.savefig('Fusion_CNNLSTMCNN_Train_error.pdf')
  pyplot.close()
  return fusion

def build_model_fusion_lstm_cnn_lstm(train_d1, n_input, n_output, ftrIndx_m1):
  
  # define parameters
  verbose, epochs, batch_size = 1, 5, 200

  #MODEL 1 - LSTM-CNN-LSTM Takes weather+electric data
  train_x_1, train_y_1 = to_supervised(train_d1, n_input, n_output, ftrIndex_m1)
  n_timesteps, n_features, n_outputs = train_x_1.shape[1], train_x_1.shape[2], train_y_1.shape[1]
  # define model
  model1 = Sequential()

  # LSTM
  model1.add(LSTM(5, activation='relu',
        return_sequences=True, 
        input_shape=(n_timesteps, n_features)))
  model1.add(Dense(n_outputs))
  ####

  model1.add(Conv1D(filters=16, kernel_size=9, activation='relu', input_shape=(n_timesteps,n_features)))
  model1.add(MaxPooling1D(pool_size=4))

  # Stacked LSTM
  model1.add(LSTM(5, activation='relu',
        #return_sequences=True, 
        input_shape=(n_timesteps, n_features)))
  model1.add(Dense(n_outputs))
  model1.add(Reshape((n_output,1)))
  ####

  #MODEL 2 - CNN - Takes only electric data
  n_features = 1
  # define model
  model2 = Sequential()
  #print("features: "+str(n_features))
  model2.add(Conv1D(filters=16, kernel_size=9, activation='relu', input_shape=(n_timesteps,n_features)))
  model2.add(MaxPooling1D(pool_size=4))
  model2.add(Flatten())
  model2.add(Dense(n_outputs))
  model2.add(Reshape((n_output,1)))

  #CONCATENATE
  merged = concatenate([model1.output, model2.output], axis=2)

  #LINEAR REGRESSION
  linReg = Dense(10, input_dim=2, kernel_initializer='normal', activation='relu')(merged)
  linReg = Dense(1, kernel_initializer='normal') (linReg)

  #COMPLETE MODEL
  fusion = Model(inputs=[model1.input, model2.input], outputs=linReg)
  plot_model(fusion, show_shapes=True, show_layer_names=False, to_file='fusion_lstm_cnn_lstm.pdf')
  fusion.compile(loss='mse', optimizer='adam')

  train_y_1 = train_y_1.reshape(train_y_1.shape[0], train_y_1.shape[1], 1)
  train_x_2 = train_x_1[0:int(train_x_1.shape[0]*3/4), :, ftrIndex_m1]
  train_x_2 = train_x_2.reshape(train_x_2.shape[0], train_x_2.shape[1], 1)
  train_x_2_val = train_x_1[int(train_x_1.shape[0]*3/4):, :, ftrIndex_m1]
  train_x_2_val = train_x_2_val.reshape(train_x_2_val.shape[0], train_x_2_val.shape[1], 1)

  # fit network
  history = fusion.fit([train_x_1[0:int(train_x_1.shape[0]*3/4)], train_x_2], train_y_1[0:int(train_y_1.shape[0]*3/4)], validation_data=([train_x_1[int(train_x_1.shape[0]*3/4):], train_x_2_val], train_y_1[int(train_y_1.shape[0]*3/4):]), epochs=epochs, batch_size=batch_size, verbose=verbose)
  #ERROR PLOT####
  pyplot.ylabel('Error')
  pyplot.xlabel('Epochs')
  #pyplot.title("CNN")
  pyplot.plot(history.history['loss'], label="Train")
  pyplot.plot(history.history['val_loss'], label="Validation")
  pyplot.legend()
  #pyplot.show()
  pyplot.savefig('Fusion_LSTMCNNLSTM_Train_error.pdf')
  pyplot.close()
  return fusion

# make a forecast for fusion model
def forecast_fusion(model, history_m1, history_m2, n_input):
  # flatten data
  input_x_m1 = array(history_m1)
  input_x_m2 = array(history_m2)

  # reshape into [1, n_input, n]
  input_x_m1 = input_x_m1.reshape((1, input_x_m1.shape[0], input_x_m1.shape[1]))
  input_x_m2 = input_x_m2.reshape((1, input_x_m2.shape[0], 1))
  yhat = model.predict([input_x_m1, input_x_m2], verbose=0)
  # we only want the vector forecast
  yhat = yhat[0]
  return yhat

# evaluate a single model
def evaluate_fusion(model, train_m1, test_m1, n_input, n_output, ftrIndex):
  # fit model
  # history is a list of weekly data
  history_m1 = [x for x in test_m1]
  # walk-forward validation over each week
  predictions = list()
  real_values = list()

  for i in range(test_m1.shape[0]-n_input-n_output+1):
    real_values.append(test_m1[i+n_input:i+n_input+n_output, ftrIndex])
    # predict the week
    yhat_sequence = forecast_fusion(model, test_m1[i:i+n_input], test_m1[i:i+n_input, ftrIndex], n_input) 
    # store the predictions
    predictions.append(yhat_sequence)
  # evaluate predictions days for each week
  real_values = array(real_values)
  predictions = array(predictions)
  predictions = predictions.reshape(predictions.shape[0],n_output)
  scores = list()

  ret1, ret2, ret3, ret4  = evaluate_forecasts(real_values, predictions)
  scores.append(ret1)
  scores.append(ret2)
  scores.append(ret3)
  scores.append(ret4)
  
  predictions = predictions.reshape(predictions.shape[0],n_output, 1)
  predictions = min_max_scaler2.inverse_transform(predictions[-1])
  return scores, predictions #mean(predictions, axis=0)

def conclusion_errors(outer_list, prdDays, modelNames, markers, type):
  # plot scores
  hours = list(range(1, prdDays+1))#, 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20', 'd21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30']
  for inner_list, modelName, marker in zip(outer_list, modelNames, markers):
    result_mse = mean([row[0] for row in inner_list[:]], axis = 0)
    pyplot.plot(hours, result_mse, marker=marker, label=modelName)
  #pyplot.title('Univariate_Models_'+type)
  pyplot.legend()
  pyplot.savefig('Fusion_Models_mse_'+type+'.pdf')
  pyplot.close()

  for inner_list, modelName, marker in zip(outer_list, modelNames, markers):
    result_rmse = mean([row[1] for row in inner_list[:]], axis = 0)
    pyplot.plot(hours, result_rmse, marker=marker, label=modelName)
  #pyplot.title('Univariate_Models_'+type)
  pyplot.legend()
  pyplot.savefig('Fusion_Models_rmse_'+type+'.pdf')
  pyplot.close()

  for inner_list, modelName, marker in zip(outer_list, modelNames, markers):
    result_mae = mean([row[2] for row in inner_list[:]], axis = 0)
    pyplot.plot(hours, result_mae, marker=marker, label=modelName)
  #pyplot.title('Univariate_Models_'+type)
  pyplot.legend()
  pyplot.savefig('Fusion_Models_mae_'+type+'.pdf')
  pyplot.close()

  for inner_list, modelName, marker in zip(outer_list, modelNames, markers):
    result_mae = mean([row[3] for row in inner_list[:]], axis = 0)
    pyplot.plot(hours, result_mae, marker=marker, label=modelName)
  #pyplot.title('Univariate_Models_'+type)
  pyplot.legend()
  pyplot.savefig('Fusion_Models_r2_'+type+'.pdf')
  pyplot.close()

def conclusion_preds(outer_list, prdDays, modelNames, markers, type):
  # plot scores
  hours = list(range(1, prdDays+1))#, 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20', 'd21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30']
  for inner_list, modelName, marker in zip(outer_list, modelNames, markers):
    result = mean(inner_list, axis = 0)
    pyplot.plot(hours, result, marker=marker, label=modelName)
  #pyplot.title('Univariate_Models_'+type)
  pyplot.legend()
  pyplot.savefig('Fusion_Models_'+type+'.pdf')
  pyplot.close()

def prepareTrainTestData(data, foldSize):
  train = []
  foldLen = (int)(len(data) / foldSize)
  i = 0
  for i in range(foldSize):
    train.append(data[:(i+1)*foldLen])
  
  #test.append(data[(int)(len(data) / 2):])
  return train


# load the new file
#dataset = read_csv('complete_data.csv', header=0, usecols=[5,6,7,8,9,10,11,12], decimal=",", delimiter=";", infer_datetime_format=True, dayfirst=True, parse_dates=['Date'], index_col=['Date'])

if os.name == 'nt':
  dataset_1 = read_csv(r'data\USA_Electric_Climate\IL_DATA\ninja_weather_country_US_IL_merra_2_population_weighted.csv', header=0, usecols=[0,1,2,3,4,5,6,7,8], infer_datetime_format=True, parse_dates=['time'], index_col=['time'])
  dataset_2 = read_csv(r'data\USA_Electric_Climate\hourly-energy-consumption\COMED_hourly.csv', header=0, usecols=[0,1], infer_datetime_format=True, parse_dates=['Datetime'], index_col=['Datetime'])
else:
  dataset_1 = read_csv(r'data/ninja_weather_country_US_IL_merra_2_population_weighted.csv', header=0, usecols=[0,1,2,3,4,5,6,7,8], infer_datetime_format=True, parse_dates=['time'], index_col=['time'])
  dataset_2 = read_csv(r'data/COMED_hourly.csv', header=0, usecols=[0,1], infer_datetime_format=True, parse_dates=['Datetime'], index_col=['Datetime'])

dataset_2 = dataset_2.astype('float32')
end_date = '12/31/2016 23:00'
start_date = '1/1/2011 0:00'
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler2 = preprocessing.MinMaxScaler()

dataset_1 = dataset_1.iloc[271752:324359]
dataset_1 = dataset_1[0:15000]

dataset_2 = dataset_2.iloc[0:52607]
dataset_2 = dataset_2[0:15000]

dataset_1 = append(dataset_1, dataset_2, axis=1)
#after this line, data starts from 2011 till 2016 scaled in [0,1]


set_option('display.width', 1000)
set_option('precision', 3)
correlations, _ = spearmanr(dataset_1)
print(correlations)
# plot correlation matrix
#fig = pyplot.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(correlations, vmin=-1, vmax=1)
#fig.colorbar(cax)
#pyplot.show()


dayPredictionErrors_fusion_cnn_lstm = list()
dayPredictionErrors_fusion_lstm_lstm = list()
dayPredictionErrors_fusion_cnn_lstm_cnn = list()
dayPredictionErrors_fusion_lstm_cnn_lstm = list()

dayPredictions_fusion_cnn_lstm = list()
dayPredictions_fusion_lstm_lstm = list()
dayPredictions_fusion_cnn_lstm_cnn = list()
dayPredictions_fusion_lstm_cnn_lstm = list()
dayPredictions_RealValues = list()

n_input = 50*24
n_output = 8*24
foldSize = 5
ftrIndex_m1, ftrIndex_m2 = 8, 0 #since working with testdata1 which is fused, then electric consumption is 3rd and on electric dataset it is 0

# split into train and test
train_d1 = prepareTrainTestData(dataset_1, foldSize)

# enumerate splits
for i in range(foldSize):
  train_d1_data, test_d1_data = [], []
  realValues = []
  if(train_d1[i].shape[0] % n_input != 0):
   train_d1_data = (train_d1[i])[0:-(train_d1[i].shape[0] % n_input),:]
  else:
   train_d1_data = train_d1[i]
 
  # get test data from last part of trains
  test_d1_data  = train_d1_data[-( n_input + n_output + 50):,:]
  train_d1_data = train_d1_data[:-(n_input + n_output + 50),:]

  j = 0
  realValues.clear()
  for j in range(len(test_d1_data)-n_input-n_output+1):
    #prepare real values
    realValues.append(test_d1_data[j+n_input:j+n_input+n_output, ftrIndex_m1])

  #realValues = array(realValues)
  dayPredictions_RealValues.append(realValues[-1])
  
  train_d1_data_excptLastCol_Scaled = min_max_scaler.fit_transform(train_d1_data[:,0:8])
  test_d1_data_excptLastCol_Scaled = min_max_scaler.fit_transform(test_d1_data[:,0:8])

  train_d1_data_last_col = train_d1_data[:,8].reshape(train_d1_data[:,8].shape[0], 1)
  train_d1_data_last_col_scaled = min_max_scaler2.fit_transform(train_d1_data_last_col)
  train_d1_data = append(train_d1_data_excptLastCol_Scaled, train_d1_data_last_col_scaled, axis=1)
  
  test_d1_data_last_col = test_d1_data[:,8].reshape(test_d1_data[:,8].shape[0], 1)
  test_d1_data_last_col_scaled = min_max_scaler2.fit_transform(test_d1_data_last_col)
  test_d1_data = append(test_d1_data_excptLastCol_Scaled, test_d1_data_last_col_scaled, axis=1)

  
  print("CNN-LSTM Hybrid started")
  fusion_cnn_lstm = build_model_fusion_cnn_lstm(train_d1_data, n_input, n_output, ftrIndex_m1)
  errors_fusion, predictions_fusion = evaluate_fusion(fusion_cnn_lstm, train_d1_data, test_d1_data, n_input, n_output, ftrIndex_m1)
  dayPredictionErrors_fusion_cnn_lstm.append(errors_fusion)
  dayPredictions_fusion_cnn_lstm.append(predictions_fusion)
  print("CNN-LSTM Hybrid ended")

  print("LSTM-LSTM Hybrid started")
  fusion_lstm_lstm = build_model_fusion_lstm_lstm(train_d1_data, n_input, n_output, ftrIndex_m1)
  errors_fusion, predictions_fusion = evaluate_fusion(fusion_lstm_lstm, train_d1_data, test_d1_data, n_input, n_output, ftrIndex_m1)
  dayPredictionErrors_fusion_lstm_lstm.append(errors_fusion)
  dayPredictions_fusion_lstm_lstm.append(predictions_fusion)
  print("LSTM-LSTM Hybrid ended")
  #
  print("CNN-LSTM-CNN Hybrid started")
  fusion_cnn_lstm_cnn = build_model_fusion_cnn_lstm_cnn(train_d1_data, n_input, n_output, ftrIndex_m1)
  errors_fusion, predictions_fusion = evaluate_fusion(fusion_cnn_lstm_cnn, train_d1_data, test_d1_data, n_input, n_output, ftrIndex_m1)
  dayPredictionErrors_fusion_cnn_lstm_cnn.append(errors_fusion)
  dayPredictions_fusion_cnn_lstm_cnn.append(predictions_fusion)
  print("CNN-LSTM-CNN Hybrid ended")
  #
  print("LSTM-CNN-LSTM Hybrid started")
  fusion_lstm_cnn_lstm = build_model_fusion_lstm_cnn_lstm(train_d1_data, n_input, n_output, ftrIndex_m1)
  errors_fusion, predictions_fusion = evaluate_fusion(fusion_lstm_cnn_lstm, train_d1_data, test_d1_data, n_input, n_output, ftrIndex_m1)
  dayPredictionErrors_fusion_lstm_cnn_lstm.append(errors_fusion)
  dayPredictions_fusion_lstm_cnn_lstm.append(predictions_fusion)
  print("LSTM-CNN-LSTM Hybrid ended")

 
markers = []
markers.append('.')
markers.append('*')
markers.append('+')
markers.append('v')

modelNames = []
modelNames.append('CNN_LSTM')
modelNames.append('LSTM_LSTM') 
modelNames.append('CNN_LSTM_CNN')
modelNames.append('LSTM_CNN_LSTM')

predictionErrors = []
predictionErrors.append(dayPredictionErrors_fusion_cnn_lstm)
predictionErrors.append(dayPredictionErrors_fusion_lstm_lstm)
predictionErrors.append(dayPredictionErrors_fusion_cnn_lstm_cnn)
predictionErrors.append(dayPredictionErrors_fusion_lstm_cnn_lstm)

conclusion_errors(predictionErrors, n_output, modelNames, markers, 'errors')

markers = []
markers.append('.')
markers.append('*')
markers.append('+')
markers.append('v')
markers.append('x')

modelNames = []
modelNames.append('CNN_LSTM')
modelNames.append('LSTM_LSTM')
modelNames.append('CNN_LSTM_CNN')
modelNames.append('LSTM_CNN_LSTM')
modelNames.append('Real Values')

predictions = []
predictions.append(dayPredictions_fusion_cnn_lstm)
predictions.append(dayPredictions_fusion_lstm_lstm)
predictions.append(dayPredictions_fusion_cnn_lstm_cnn)
predictions.append(dayPredictions_fusion_lstm_cnn_lstm)
predictions.append(dayPredictions_RealValues)

conclusion_preds(predictions, n_output, modelNames, markers, 'predictions')