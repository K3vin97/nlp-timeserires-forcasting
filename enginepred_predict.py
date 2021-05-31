'''
Created on 11/01/2018

@author: Francesco Pugliese
'''

# Preprocessing
from Preprocessing.preprocessing import load_pollution_datasets, load_bitcoin_datasets, load_bitcoin_from_db, load_bitcoin_from_db_for_prediction, save_bitcoin_into_db
from Settings.settings import SetParameters

# Models imports
from Models.kpred_model_sae import PredModelSAE
from Models.kpred_model_lstm import PredModelLSTM 
from Models.kpred_model_gru import PredModelGRU 

# Visualization
from View.view import plot_dataset, plot_loss, plot_predictions, print_computation_time_in_days

# Miscellaneous
from Misc.metrics import mean_absolute_percentage_error
from Misc.utils import set_cpu_or_gpu

# Predictions
from Prediction.prediction import single_prediction

# Other imports
import os
import sys
import numpy
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import Series
from pandas import errors
import timeit

import pdb
import time

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

#date_time = time.strftime('%Y-%m-%d %H:%M:%S')
date_time = time.strftime('%Y-%m-%d')
global_start_time = timeit.default_timer()

# Read the Configuration File
parameters = SetParameters.init_by_config_file()

# Set the CPU or GPU
set_cpu_or_gpu(parameters)

# Visualization options for the training
if parameters.verbose == False:
    summary = False
    verbose_fit = 0
    verbose_model_check_point = 0
else: 
    summary = True
    verbose_fit = 2
    verbose_model_check_point = 1

# Loop onto all the destination crypto coins
if parameters.loop_on_crypto_dest_type == True and parameters.input_from_db == True: 
    crypto_dest_type_possible_intervals_list = parameters.crypto_dest_type_possible_intervals.split(',')
    total_crypto_dest = len(crypto_dest_type_possible_intervals_list)
else: 
    total_crypto_dest = 1
    crypto_dest_type = parameters.crypto_dest_type

# Loop onto all the sampling intervals
if parameters.loop_on_dataset_sampling_intervals == True: 
    dataset_sampling_loop_intervals_list = parameters.dataset_sampling_loop_intervals.split(',')
    total_dataset_sampling_loop_intervals = len(dataset_sampling_loop_intervals_list)
else: 
    total_dataset_sampling_loop_intervals = 1
    dataset_sampling_interval = parameters.dataset_sampling_interval

if parameters.export_metrics_into_csv == True:
    file_metrics_name = parameters.output_path+'/'+parameters.neural_model+"_"+date_time+"_T"+str(parameters.T)+"_"+parameters.metrics_file
    open(file_metrics_name, 'w').close()
    file_metrics = open(file_metrics_name, 'ab')
    try: 
        numpy.savetxt(file_metrics, [], header = "Origin Crypto Coin, Destination Crypto Coin, T, Dataset Sampling Interval, RMSE, MAPE", delimiter = ",", fmt = "%s")
    except UnicodeEncodeError: 
        print("\nExport of Metrics Error.\n")
    
for si in range(total_dataset_sampling_loop_intervals): 
    training_start_time = timeit.default_timer()
  
    for cd in range(total_crypto_dest): 
    
        if parameters.loop_on_crypto_dest_type == True and parameters.input_from_db == True: 
            crypto_dest_type = crypto_dest_type_possible_intervals_list[cd] 
    
        if parameters.loop_on_dataset_sampling_intervals == True: 
            dataset_sampling_interval = dataset_sampling_loop_intervals_list[si] 

        if parameters.verbose == False: 
            print('\nPredicting on: ', parameters.crypto_orig_type+'/'+crypto_dest_type)

        if parameters.stateful == True:
            stateful = "_stateful_"
        else: 
            stateful = ""
            
        files_prefix = parameters.crypto_orig_type+"-"+crypto_dest_type+"_T"+str(parameters.T)+"_"+dataset_sampling_interval+"_"+parameters.neural_model+"_"		# default prefix of each saved file
        if parameters.full_dataset == True:
            files_prefix = files_prefix + "full_" 
        #pdb.set_trace()
        if parameters.stateful == True:
            files_prefix = files_prefix + "stateful_" 

        str_header1 = "Date-Time, Predicted "+parameters.crypto_orig_type+"/"+crypto_dest_type+" Value, Real "+parameters.crypto_orig_type+"/"+crypto_dest_type+" Value"		# header for the ouput file
        str_header2 = "Predicted "+parameters.crypto_orig_type+"/"+crypto_dest_type+" Value, Real "+parameters.crypto_orig_type+"/"+crypto_dest_type+" Value"		# header for the ouput file

        if parameters.pollution_benchmark == True:
            train_X, train_y, test_X, test_y, scaler = load_pollution_datasets(parameters.pollution_dataset_path, parameters.pollution_dataset_file, parameters.prepare_pollution_dataset, parameters.pollution_dataset_file_raw, parameters.output_size, parameters.verbose)
            pred_dataset_path = parameters.pollution_dataset_path
            pred_dataset_file = parameters.pollution_dataset_file
        elif parameters.bitcoin == True: 
            if parameters.input_from_db == True:
                if parameters.prediction_data_metrics_evaluation == True:     
                    train_X, train_y, valid_X, valid_y, test_X, test_y, test_no_labels, scaler, train_dates, test_dates, input_dim, selected_input_dim = load_bitcoin_from_db(parameters.db_connection_sequence, parameters.crypto_orig_type, crypto_dest_type, parameters.training_set_n_years, parameters.training_set_percentage, dataset_sampling_interval, parameters.dataset_sampling_possible_intervals, parameters.full_dataset, parameters.T, parameters.output_size, parameters.normalize, parameters.input_columns, parameters.n_sentiment_columns, parameters.output_columns, parameters.export_input_db_into_csv, parameters.output_path+'/'+files_prefix+parameters.input_db_into_csv_file, parameters.verbose)
                else: 
                    test_X, test_y, test_no_labels, scaler, test_dates, input_dim, selected_input_dim = load_bitcoin_from_db_for_prediction(parameters.db_connection_sequence, parameters.crypto_orig_type, crypto_dest_type, parameters.prediction_data_size, dataset_sampling_interval, parameters.dataset_sampling_possible_intervals, parameters.full_dataset, parameters.T, parameters.output_size, parameters.normalize, parameters.input_columns, parameters.output_columns, parameters.export_input_db_into_csv, parameters.output_path+'/'+files_prefix+parameters.input_db_into_csv_file, parameters.verbose)
            else:
                # do not include sentiment with load from csv datasets, only from db
                train_X, train_y, valid_X, valid_y, test_X, test_y, test_no_labels, scaler, input_dim, selected_input_dim = load_bitcoin_datasets(parameters.bitcoin_dataset_path, parameters.bitcoin_dataset_file, parameters.training_set_n_years, parameters.training_set_percentage, dataset_sampling_interval, parameters.dataset_sampling_possible_intervals, parameters.full_dataset, parameters.T, parameters.output_size, parameters.normalize, parameters.input_columns, parameters.output_columns, parameters.verbose, parameters.remove_first_column, parameters.skip_rounding)
                pred_dataset_path = parameters.bitcoin_dataset_path
                pred_dataset_file = parameters.bitcoin_dataset_file
            
            n_residual_elements = len(test_no_labels)               # determines the number of "no label" elements, useful for shif the dates during the savings operations
        if parameters.prediction_data_metrics_evaluation == False: 
            train_X = test_X
        
        if parameters.neural_model == 'sae_lstm':
            models, sae_model = PredModelSAE.build(input_length=train_X.shape[1], vector_dim=train_X.shape[2], output_size = parameters.output_size, depth = parameters.sae_depth, encoding_dim = parameters.sae_encoding_dim, summary=True)	
            old_test_X = test_X
            test = test_X.reshape((test_X.shape[0], test_X.shape[2]))
            test_nl = test_no_labels.reshape((test_no_labels.shape[0], test_no_labels.shape[2]))
            
            # Load the saved best stacked autoencoder
            if os.path.isfile(parameters.models_path+'/'+files_prefix+"sae_part_"+parameters.model_file):
                models[parameters.sae_depth - 1].load_weights(parameters.models_path+'/'+files_prefix+"sae_part_"+parameters.model_file) 
            else:
                print('\nPre-trained stacked autoencoder not found: %s.' % (parameters.models_path+'/'+files_prefix+"sae_part_"+parameters.model_file))
                sys.exit("")

            test_X = sae_model.predict(test)
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

            if parameters.T <= 1: 
                old_test_no_labels = test_no_labels
                test_no_labels = sae_model.predict(test_nl)
                test_no_labels = test_no_labels.reshape((test_no_labels.shape[0], 1, test_no_labels.shape[1]))
        
        if parameters.stateful == True:
            train_off_set = (train_X.shape[0] % parameters.batch_size)  
            train_X = train_X[train_off_set:]
            train_y = train_y[train_off_set:]
            valid_off_set = (valid_X.shape[0] % parameters.batch_size)  
            valid_X = valid_X[valid_off_set:]
            valid_y = valid_y[valid_off_set:]
            test_off_set = (test_X.shape[0] % parameters.batch_size)  
            test_X = test_X[test_off_set:]
            test_y = test_y[test_off_set:]

        if parameters.neural_model == 'lstm' or parameters.neural_model == 'sae_lstm':
            deepnetwork = PredModelLSTM.build(input_length=test_X.shape[1], vector_dim=test_X.shape[2], output_size = parameters.output_size, batch_size = parameters.batch_size, n_hiddens = parameters.n_hiddens, summary=summary, stateful = parameters.stateful)	

        if parameters.neural_model == 'gru':
            deepnetwork = PredModelGRU.build(input_length=test_X.shape[1], vector_dim=test_X.shape[2], output_size = parameters.output_size, batch_size = parameters.batch_size, n_hiddens = parameters.n_hiddens, summary=summary, stateful = parameters.stateful)	

        deepnetwork.compile(loss='mae', optimizer='adam')

        if os.path.isfile(parameters.models_path+'/'+files_prefix+parameters.model_file):
            deepnetwork.load_weights(parameters.models_path+'/'+files_prefix+parameters.model_file) 
        else:
            print('\nPre-trained model not found: %s.' % (parameters.models_path+'/'+files_prefix+parameters.model_file))
            sys.exit("")
           
        if parameters.prediction_data_metrics_evaluation == True: 
            if parameters.dataset_view == True and OS != "Linux" and parameters.input_from_db == False:
                plot_dataset(pred_dataset_path, pred_dataset_file)

            # load history
            try: 
                history = read_csv(parameters.log_path+'/'+files_prefix+parameters.log_file, header=0)
                # plot history
                if parameters.history_view == True and OS != "Linux":
                    plot_loss(history)
            except errors.EmptyDataError: 
                print("\nHistory file not readable.\n")
                history = None

        # make a prediction on test set with labels
        yhat = deepnetwork.predict(test_X, batch_size=parameters.batch_size)                  # prediction of y values
        if parameters.neural_model == 'sae_lstm':
            test_X = old_test_X 

        if parameters.normalize == True: 
            test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        	# invert scaling for forecast
            #inv_yhat = concatenate((test_X[:, 0:(int(parameters.output_columns)-1)], yhat, test_X[:, int(parameters.output_columns):selected_input_dim+1]), axis=1)
            inv_yhat = concatenate((test_X[:, 0:(int(parameters.output_columns)-1)], yhat, test_X[:, int(parameters.output_columns):selected_input_dim]), axis=1)
            if parameters.input_from_db == True: 
                inv_yhat = numpy.hstack((inv_yhat, numpy.zeros((len(inv_yhat),input_dim-selected_input_dim))))  	    # add columns of 0 up to the end of the array
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,(int(parameters.output_columns)-1)]
            if parameters.prediction_data_metrics_evaluation == True: 
                # invert scaling for actual
                test_y = test_y.reshape((len(test_y), parameters.output_size))
                #inv_y = concatenate((test_X[:, 0:(int(parameters.output_columns)-1)], test_y, test_X[:, int(parameters.output_columns):selected_input_dim+1]), axis=1)
                inv_y = concatenate((test_X[:, 0:(int(parameters.output_columns)-1)], test_y, test_X[:, int(parameters.output_columns):selected_input_dim]), axis=1)
                if parameters.input_from_db == True: 
                    inv_y = numpy.hstack((inv_y, numpy.zeros((len(inv_y),input_dim-selected_input_dim))))  	                # add columns of 0 up to the end of the array
                inv_y = scaler.inverse_transform(inv_y)
                inv_y = inv_y[:,(int(parameters.output_columns)-1)]
                # calculate RMSE
                rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
                mape = mean_absolute_percentage_error(inv_y, inv_yhat)
        else: 
            if parameters.prediction_data_metrics_evaluation == True: 
                rmse = sqrt(mean_squared_error(test_y, yhat))
                mape = mean_absolute_percentage_error(test_y, inv_yhat)
            
        if parameters.prediction_data_metrics_evaluation == True: 
            print('\n\nTest RMSE: %f' % rmse)
            print('Test MAPE: %f' % mape)
            if history is not None and parameters.verbose == True: 
                print('Minimum Validation Loss: %f ' % (numpy.min(history['val_loss'])))
                print('Minimum Validation Loss Epoch: %i ' % (Series.idxmin(history['val_loss'])+1))

        # make a prediction on test set without labels
        if parameters.T <= 1 and len(test_no_labels)>=parameters.batch_size: 
            yhat_no_labels = deepnetwork.predict(test_no_labels, batch_size=parameters.batch_size)
            
            if parameters.neural_model == 'sae_lstm':
                test_no_labels = old_test_no_labels
            if parameters.normalize == True: 
                test_no_labels = test_no_labels.reshape((test_no_labels.shape[0], test_no_labels.shape[2]))
                # invert scaling for forecast
                inv_yhat_no_labels = concatenate((test_no_labels[:, 0:(int(parameters.output_columns)-1)], yhat_no_labels, test_no_labels[:, int(parameters.output_columns):selected_input_dim+1]), axis=1)
                if parameters.input_from_db == True: 
                    inv_yhat_no_labels = numpy.hstack((inv_yhat_no_labels, numpy.zeros((len(inv_yhat_no_labels),input_dim-selected_input_dim))))  	    # add columns of 0 up to the end of the array
                inv_yhat_no_labels = scaler.inverse_transform(inv_yhat_no_labels)
                inv_yhat_no_labels = inv_yhat_no_labels[:,(int(parameters.output_columns)-1)]
            
            inv_yhat = numpy.hstack((inv_yhat, inv_yhat_no_labels))                  # add elements with no labels
        
        if parameters.export_metrics_into_csv == True and parameters.prediction_data_metrics_evaluation == True:
            output = [parameters.crypto_orig_type+", "+crypto_dest_type+", "+str(parameters.T)+", "+str(dataset_sampling_interval)+", "+str(rmse)+", "+str(mape)]
            
            try: 
                numpy.savetxt(file_metrics, output, delimiter = ",", fmt = "%s")
            except UnicodeEncodeError: 
                print("\nExport of Metrics Error.\n")

        if parameters.export_predictions_into_csv == True and parameters.prediction_data_metrics_evaluation == True:
            if parameters.T <= 1 and len(test_no_labels)>=parameters.batch_size: 
                none_arr = numpy.asarray([None] * inv_yhat_no_labels.shape[0])           # create an empty piece of array
                inv_y = numpy.hstack((inv_y, none_arr))                                  # add a certain number of None values within the real y which are not available
            if parameters.input_from_db == True:
                str_test_dates = []
                for test_date in test_dates: 
                    str_test_dates.append(test_date.strftime('%Y-%m-%d %H:%M:%S'))
    
                str_test_dates = numpy.asarray(str_test_dates[:len(str_test_dates)]).reshape((-1,1))
            inv_yhat = inv_yhat.reshape((-1,1))
            inv_y = inv_y.reshape((-1,1))
            
            if parameters.input_from_db == True:
                if len(str_test_dates)>len(inv_y):
                    str_test_dates = str_test_dates[0:len(str_test_dates)-(len(str_test_dates)-len(inv_y))] # shorten str_test_dates
                output = numpy.hstack((str_test_dates, inv_yhat, inv_y))
                str_header = str_header1
            else: 
                output = numpy.hstack((inv_yhat, inv_y))                        # with no dates
                str_header = str_header2
            try: 
                numpy.savetxt(parameters.output_path+'/'+files_prefix+parameters.output_file, output, header = str_header, delimiter = ",", fmt = "%s")
            except UnicodeEncodeError: 
                print("\nExport of Predictions Error.\n")

        if parameters.export_predictions_into_db == True and parameters.input_from_db == True: 
            save_bitcoin_into_db(parameters.db_connection_sequence, parameters.crypto_orig_type, crypto_dest_type, test_dates, parameters.sampling_basic_unit, n_residual_elements, inv_yhat, dataset_sampling_interval, parameters.dataset_sampling_possible_intervals, parameters.full_dataset, parameters.verbose)               								  # Save outcomes into the DB 

        if parameters.prediction_data_metrics_evaluation == True: 
            if parameters.predictions_view == True and parameters.OS != "Linux":
                plot_predictions(parameters.output_path+'/'+files_prefix+parameters.predictions_view_file, rmse, mape, inv_yhat, inv_y, False, 5)

if parameters.press_key == True: 
    input = input("\n\nPress a key to end...")

if parameters.prediction_data_metrics_evaluation == True: 
    if parameters.export_metrics_into_csv == True:
        file_metrics.close()

global_end_time = timeit.default_timer()

global_time_in_sec = global_end_time - global_start_time
print ('\n\n')
print_computation_time_in_days(global_time_in_sec)                  # displays computation time in days, hours, etc
print ('\n\n')
