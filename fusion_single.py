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
from numpy import isnan
from statsmodels.tsa.stattools import adfuller
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.ensemble import RandomForestRegressor
#import plaidml.keras
#plaidml.keras.install_backend()

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input, Flatten, merge, Reshape, Dropout, Masking, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import MaxPooling1D
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import TerminateOnNaN
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
  # calculate an RMSE score for each day
  for i in range(predicted.shape[1]): 
      # calculate mse
      if(actual.ndim != predicted.ndim):
        actual = actual.reshape(actual.shape[0], actual.shape[1]*actual.shape[2])
      try:
        mse = mean_squared_error(actual[:, i], array([abs(ele) for ele in predicted[:, i]]))
        mae = mean_absolute_error(actual[:, i], array([abs(ele) for ele in predicted[:, i]]))
        rmse = sqrt(mse)
        # store
        scores_rmse.append(rmse)
        scores_mse.append(mse)
        scores_mae.append(mae)
      except:
        mse = 0
        mae = 0
        rmse = 0
 
  return scores_mse, scores_rmse, scores_mae

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

def build_model_cnn(train, n_input, n_output, ftrIndex):
  # prepare data
  print("build_model::train "+str(train.shape))
  train_x, train_y = to_supervised(train, n_input, n_output, ftrIndex)
  # define parameters
  verbose, epochs, batch_size = 1, 64, 128
  n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
  # define model
  model = Sequential()
  model.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Conv1D(filters=5, kernel_size=3, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Conv1D(filters=3, kernel_size=2, activation='relu'))

  model.add(Flatten())
  model.add(Dense(n_outputs))

  model.compile(loss='mse', optimizer='adam')
  plot_model(model, show_shapes=True, show_layer_names=False, to_file='model_cnn_single.pdf')
  #pyplot.show()
  # fit network
  model.fit(train_x[0:int(train_x.shape[0]*3/4)], train_y[0:int(train_x.shape[0]*3/4)], 
             epochs=epochs, validation_data=(train_x[int(train_x.shape[0]*3/4):], train_y[int(train_x.shape[0]*3/4):]), 
             batch_size=batch_size, callbacks=[TerminateOnNaN(), EarlyStopping(monitor='loss', patience=3)], shuffle=False, verbose=verbose)
  
  return model

def build_model_lstm(train, n_input, n_output, ftrIndex):
  # prepare data
  print("build_model::train "+str(train.shape))
  train_x, train_y = to_supervised(train, n_input, n_output, ftrIndex)
  # define parameters
  verbose, epochs, batch_size = 1, 64, 128
  n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
  
  # define model
  model = Sequential()
  model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
  model.add(LSTM(100, activation='relu'))

  model.add(Dense(n_outputs))
  
  model.compile(loss='mse', optimizer='adam')
  plot_model(model, show_shapes=True, show_layer_names=False, to_file='model_lstm_arch.pdf')
  
  model.fit(train_x[0:int(train_x.shape[0]*3/4)], train_y[0:int(train_x.shape[0]*3/4)], 
            epochs=epochs, validation_data=(train_x[int(train_x.shape[0]*3/4):], train_y[int(train_x.shape[0]*3/4):]), 
            batch_size=batch_size, callbacks=[TerminateOnNaN(), EarlyStopping(monitor='loss', patience=3)], shuffle=False, verbose=verbose)
 
  return model

def build_model_rf(train, n_input, n_output, ftrIndex):
  # prepare data
  train_x, train_y = to_supervised(train, n_input, n_output, ftrIndex)
  train_x = train_x.reshape(train_x.shape[0], train_x.shape[1])
  #train_y = train_y.reshape(1, train_y.shape[0])
  # Train the model on training data
  # define model
  rf = RandomForestRegressor(n_estimators = 25)
  return rf.fit(train_x, train_y)

def build_model_arima(train, n_input, n_output, ftrIndex):
  # define model
  model = ARIMA(train, order=(6,2,1))
  return model.fit()

# make a forecast for simple model
def forecast(model, data, n_input):
  # flatten data
  input_x = array(data)
  # forecast the next week
  input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
  yhat = model.predict(input_x, verbose=0)
  yhat = yhat[0]

  return yhat

# make a forecast for fusion model
def forecast_fusion(model, history_m1, history_m2, n_input):
  # flatten data
  data_m1 = array(history_m1)
  data_m2 = array(history_m2)
  #data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
  # retrieve last observations for input data
  input_x_m1 = data_m1[-n_input:, :]
  input_x_m2 = data_m2[-n_input:, :]
  # forecast the next week
  # reshape into [1, n_input, n]
  input_x_m1 = input_x_m1.reshape((1, input_x_m1.shape[0], input_x_m1.shape[1]))
  input_x_m2 = input_x_m2.reshape((1, input_x_m2.shape[0], input_x_m2.shape[1]))
  yhat = model.predict([input_x_m1, input_x_m2], verbose=0)
  # we only want the vector forecast
  yhat = yhat[0]
  return yhat

# evaluate a single model
#modelIndex = 0 ->deepmodel, 1 -> ARIMA  , 2 -> VAR, 3->RF
def evaluate_model(model, train, test, n_input, n_output, ftrIndex, modelIndex):
  # fit model
  # history is a list of weekly data
  #history = [x for x in train]
  history = [x for x in test]
  # walk-forward validation over each week
  predictions = list()
  real_values = list()
  for i in range(test.shape[0]-n_input-n_output+1):
    real_values.append(test[i+n_input:i+n_input+n_output])
    if modelIndex == 0:
      # predict the week
      yhat = forecast(model, history[i:i+n_input], n_input)
      # store the predictions
      predictions.append(yhat)
    elif modelIndex == 1:
      yhat = model.forecast(steps=i+n_input+n_output) # returns [0]: forecasts, [1]: errors, [2]: confidence interval
      predictions.append(yhat[0][i+n_input:])
    elif modelIndex == 2:
      model = VAR(train)
      model_fit = model.fit()
      yhat = model_fit.forecast(model_fit.y, steps=n_output) 
      yhat = yhat[:,8]
      predictions.append(yhat)
      break
    elif modelIndex == 3:  
      input_rf = test[i:i+n_input].reshape(1, test[i:i+n_input].shape[0])
      yhat = model.predict(input_rf)  #forecast(model_fit.y, steps=n_output) 
      predictions.append(yhat[0])

    # evaluate predictions days for each week
  predictions = array(predictions)
  real_values = array(real_values)
  scores = list()

  ret1, ret2, ret3 = evaluate_forecasts(real_values, predictions)
  scores.append(ret1)
  scores.append(ret2)
  scores.append(ret3)
  predictions = min_max_scaler.inverse_transform(predictions)
  return scores, predictions[-1] #mean(predictions, axis=0)

def conclusion_errors(outer_list, prdDays, modelNames, markers, type):
  # plot scores
  hours = list(range(1, prdDays+1))#, 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20', 'd21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30']
  for inner_list, modelName, marker in zip(outer_list, modelNames, markers):
    result_mse = mean([row[0] for row in inner_list[:]], axis = 0)
    pyplot.plot(hours, result_mse, marker=marker, label=modelName)
  pyplot.legend()
  pyplot.savefig('Univariate_Models_mse_'+type+'.pdf')
  pyplot.close()

  for inner_list, modelName, marker in zip(outer_list, modelNames, markers):
    result_rmse = mean([row[1] for row in inner_list[:]], axis = 0)
    pyplot.plot(hours, result_rmse, marker=marker, label=modelName)
  pyplot.legend()
  pyplot.savefig('Univariate_Models_rmse_'+type+'.pdf')
  pyplot.close()

  for inner_list, modelName, marker in zip(outer_list, modelNames, markers):
    result_mae = mean([row[2] for row in inner_list[:]], axis = 0)
    pyplot.plot(hours, result_mae, marker=marker, label=modelName)
  pyplot.legend()
  pyplot.savefig('Univariate_Models_mae_'+type+'.pdf')
  pyplot.close()

def conclusion_preds(outer_list, prdDays, modelNames, markers, type):
  # plot scores
  hours = list(range(1, prdDays+1))#, 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20', 'd21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30']
  for inner_list, modelName, marker in zip(outer_list, modelNames, markers):
    result = mean(inner_list, axis = 0)
    pyplot.plot(hours, result, marker=marker, label=modelName)
  pyplot.legend()
  pyplot.savefig('Univariate_Models_'+type+'.pdf')
  pyplot.close()

def prepareTrainTestData(data, foldSize):
  train = []
  foldLen = (int)(len(data) / foldSize)
  i = 0
  for i in range(foldSize):
    train.append(data[:(i+1)*foldLen])
  
  return train

dataset_2 = read_csv(r'COMED_hourly.csv', header=0, usecols=[0,1], infer_datetime_format=True, parse_dates=['Datetime'], index_col=['Datetime'])

dataset_2 = dataset_2.astype('float32')
end_date = '12/31/2016 23:00'
start_date = '1/1/2011 0:00'
min_max_scaler = preprocessing.MinMaxScaler()


#X = dataset_2.values
#X = X.reshape(X.shape[0]*X.shape[1])
#result = adfuller(X)
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#  print('\t%s: %.3f' % (key, value))


dataset_2 = dataset_2.iloc[0:52607]
#dataset_2 = dataset_2[0:15000]
dataset_2 = array(dataset_2)


dayPredictionErrors_CnnOnSingle = list()
dayPredictionErrors_LstmOnSingle = list()
dayPredictionErrors_RfOnSingle = list()
dayPredictionErrors_ArOnSingle = list()
dayPredictionErrors_CnnLstmOnSingle = list()

dayPredictions_CnnOnSingle = list()
dayPredictions_LstmOnSingle = list()
dayPredictions_RfOnSingle = list()
dayPredictions_ArOnSingle = list()
dayPredictions_CnnLstmOnSingle = list()
dayPredictions_RealValues = list()

n_input = 50*24
n_output = 5*24
foldSize = 5
ftrIndex_m2 = 0 #since working with testdata1 which is fused, then electric consumption is 3rd and on electric dataset it is 0
# split into train and test
train_d2 = prepareTrainTestData(dataset_2, foldSize)

# enumerate splits
for i in range(foldSize):
  train_d2_data, test_d2_data = [], []
  realValues_d2 = []
  if(train_d2[i].shape[0] % n_input != 0):
   train_d2_data = (train_d2[i])[0:-(train_d2[i].shape[0] % n_input),:]
  else:
   train_d2_data = train_d2[i] 
 
  # get test data from last part of trains
  test_d2_data  = train_d2_data[-( n_input + n_output + 50):,:]
  train_d2_data = train_d2_data[:-(n_input + n_output + 50),:]

  
  j = 0
  realValues_d2.clear()
  for j in range(len(test_d2_data)-n_input-n_output+1):
    #prepare real values
    realValues_d2.append(test_d2_data[j+n_input:j+n_input+n_output, ftrIndex_m2])
  
  dayPredictions_RealValues.append(realValues_d2[-1])

  train_d2_data = min_max_scaler.fit_transform(train_d2_data)
  test_d2_data = min_max_scaler.fit_transform(test_d2_data)

  print("*******CNN SINGLE STARTED*****")
  model_cnn = build_model_cnn(train_d2_data, n_input, n_output, ftrIndex_m2)
  errors_cnn, predictions_cnn = evaluate_model(model_cnn, train_d2_data, test_d2_data, n_input, n_output, ftrIndex_m2, 0)
  dayPredictionErrors_CnnOnSingle.append(errors_cnn)
  dayPredictions_CnnOnSingle.append(predictions_cnn)
  print("*******CNN SINGLE ENDED*****")
  
  print("*******LSTM SINGLE STARTED*****")
  model_lstm = build_model_lstm(train_d2_data, n_input, n_output, ftrIndex_m2)
  errors_lstm, predictions_lstm = evaluate_model(model_lstm, train_d2_data, test_d2_data, n_input, n_output, ftrIndex_m2, 0)
  dayPredictionErrors_LstmOnSingle.append(errors_lstm)
  dayPredictions_LstmOnSingle.append(predictions_lstm)
  print("*******LSTM SINGLE ENDED*****")
  
  print("*******ARIMA SINGLE STARTED*****")
  arima_model = build_model_arima(train_d2_data, n_input, n_output, ftrIndex_m2)
  errors_arima, predictions_arima = evaluate_model(arima_model, train_d2_data, test_d2_data, n_input, n_output, ftrIndex_m2, 1)
  dayPredictionErrors_ArOnSingle.append(errors_arima)
  dayPredictions_ArOnSingle.append(predictions_arima)
  print("*******ARIMA SINGLE ENDED*****")
#
  print("*******RF SINGLE STARTED*****")
  rf_model = build_model_rf(train_d2_data, n_input, n_output, ftrIndex_m2)
  errors_rf, predictions_rf = evaluate_model(rf_model, train_d2_data, test_d2_data, n_input, n_output, ftrIndex_m2, 3)
  dayPredictionErrors_RfOnSingle.append(errors_rf)
  dayPredictions_RfOnSingle.append(predictions_rf)
  print("*******RF SINGLE ENDED*****")

  #print("*******CNN_LSTM SINGLE STARTED*****")
  #errors = array([None for _ in range(4)])
  #for i in range(4):
  #  errors[i] = append(errors_cnn[i][0:int(n_output/2+25)], errors_lstm[i][int(n_output/2+25):n_output], axis=0)
  #  
  #errors_2 = array(errors)
  #dayPredictionErrors_CnnLstmOnSingle.append(errors_2)
  #dayPredictions_CnnLstmOnSingle.append(append(predictions_cnn[0:int(n_output/2+15)], predictions_lstm[int(n_output/2+15):n_output], axis=0) )
  #print("*******CNN_LSTM SINGLE ENDED*****")

 
markers = []
markers.append('.')
markers.append('*')
markers.append('+')
markers.append('v')
#markers.append('x')

modelNames = []
modelNames.append('CNN')
modelNames.append('LSTM')  
modelNames.append('Arima')
modelNames.append('Random Forest')
#modelNames.append('CNN_LSTM') 

predictionErrors = []
predictionErrors.append(dayPredictionErrors_CnnOnSingle)
predictionErrors.append(dayPredictionErrors_LstmOnSingle)
predictionErrors.append(dayPredictionErrors_ArOnSingle)
predictionErrors.append(dayPredictionErrors_RfOnSingle)
#predictionErrors.append(dayPredictionErrors_CnnLstmOnSingle)

conclusion_errors(predictionErrors, n_output, modelNames, markers, 'errors')

markers = []
markers.append('.')
markers.append('*')
markers.append('+')
markers.append('v')
markers.append('x')
#markers.append(',')

modelNames = []
modelNames.append('CNN')
modelNames.append('LSTM') 
modelNames.append('Arima')
modelNames.append('Random Forest')
#modelNames.append('CNN_LSTM')
modelNames.append('Real Values')

predictions = []
predictions.append(dayPredictions_CnnOnSingle)
predictions.append(dayPredictions_LstmOnSingle)
predictions.append(dayPredictions_ArOnSingle)
predictions.append(dayPredictions_RfOnSingle)
#predictions.append(dayPredictions_CnnLstmOnSingle)
predictions.append(dayPredictions_RealValues)

conclusion_preds(predictions, n_output, modelNames, markers, 'predictions')