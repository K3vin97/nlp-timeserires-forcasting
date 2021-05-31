'''
Created on 11/01/2018

@author: Francesco Pugliese
'''

# Keras imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Miscellaneous
from Misc.utils import n_train_elements_selector, n_out_selector

# DB imports
import pymysql

# Other imports
import numpy
import os
import sys
from os import listdir
from os.path import isfile, isdir, join
import pdb
import math
import timeit

import scipy.misc
import matplotlib.pyplot as plt

from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime, timedelta

def pollution_prepare_data(pred_dataset_path='', pred_dataset_file_in='', pred_dataset_file_out=''):
    # load data
    def parse(x):
	    return datetime.strptime(x, '%Y %m %d %H')
    dataset = read_csv(pred_dataset_path+'/'+pred_dataset_file_in,  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv(pred_dataset_path+'/'+pred_dataset_file_out)
	
# convert series to supervised learning
def series_to_supervised(data, input_dim, selected_input_dim, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    res_cols = list()
    for i in range(n_out, 0, -1):                       # extracts residual rows (with no labels)
        res_cols.append(agg.ix[agg.shape[0]-(i-1),agg.shape[1]-selected_input_dim:agg.shape[1]].values)
    res_cols = numpy.vstack(res_cols)

    return [agg, res_cols]
	
# remove unuseful columns and split dataset into train and test set
def drop_columns_and_split(values, input_dim, selected_input_dim, output_size, input_columns, output_columns, training_set_n_years, training_set_percentage, dataset_sampling_interval, dataset_sampling_possible_intervals_list, full_dataset, T, normalize, verbose, skip_rounding, prediction_data_metrics_evaluation):
    
    n_out = n_out_selector(full_dataset, dataset_sampling_interval, dataset_sampling_possible_intervals_list)   # Selector of number of outputs
    
    input_columns_list = list(map(int, input_columns.split(',')))
    output_columns_list = list(map(int, output_columns.split(',')))

    if len(input_columns_list) > input_dim or len(output_columns_list) != output_size:
        print("\nNumber of selected columns exceeds the number of columnus within the dataset")
        sys.exit("")

    # Complex process and selection all the columns required		
    columns_range = list(range(1,input_dim+1))
    input_columns_drop = numpy.asarray(list(set(columns_range)-set(input_columns_list)))-1
    input_columns_drop_list = []
    for t in range(1, T+1):
        input_columns_drop_list.append(input_columns_drop+input_dim*(t-1))
    input_columns_drop = numpy.hstack(input_columns_drop_list)
	
    output_columns_drop = numpy.asarray(list(set(columns_range)-set(output_columns_list)))
    output_columns_drop = output_columns_drop+input_dim-1+input_dim*(T-1)
    output_columns_drop_list = []
    for o in range(1, n_out+1):
        output_columns_drop_list.append(output_columns_drop+input_dim*(o-1))
    output_columns_drop = numpy.hstack(output_columns_drop_list)
    
    if len(input_columns_drop) == 0:
        columns_drop = output_columns_drop
    elif len(output_columns_drop) == 0:
        columns_drop = input_columns_drop
    else: 
        columns_drop = numpy.hstack((input_columns_drop, output_columns_drop))
	
    # integer encode direction
    encoder = LabelEncoder()
    values[:,output_columns_list[0]] = encoder.fit_transform(values[:,output_columns_list[0]]) 				#one single output at moment

    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    if normalize == True: 
	    scaled = scaler.fit_transform(values)
    else: 
	    scaled = values
        
    # frame as supervised learning
    [reframed, res_cols] = series_to_supervised(scaled, input_dim, selected_input_dim, T, n_out)
    reframed.drop(reframed.columns[columns_drop], axis=1, inplace=True)
    # Removes further intermediate columns for full_dataset mode
    if full_dataset == True:
        for o_t in range(1, n_out):                       
            reframed.drop(reframed.columns[reframed.shape[1] - 2], axis=1, inplace=True)
        
    if len(res_cols) != 0:                                      # removes columns from residual array too
        numpy.delete(res_cols, columns_drop, 1)
            
    if verbose == True: 
        print(reframed.head())

    # split into train and test sets
    values = reframed.values

    if prediction_data_metrics_evaluation == True: 
        n_train_elements = n_train_elements_selector(values, training_set_n_years, training_set_percentage, dataset_sampling_interval, dataset_sampling_possible_intervals_list, skip_rounding)

        
        n_valid_test_elements = int((len(values) - n_train_elements) / 2)                           # splits into 2 parts the rest and create the validation and test set
        
        train = values[:n_train_elements, :]
        valid = values[n_train_elements:n_train_elements+n_valid_test_elements, :]
        test = values[n_train_elements+n_valid_test_elements:, :]

        test_no_labels = res_cols
        
        return [train, valid, test, test_no_labels, scaler, n_train_elements, n_valid_test_elements]
    else: 
        test = values
        test_no_labels = res_cols
        return [test, test_no_labels, scaler]
    
    
# repeat each value of a column sentiment into all the values later on til the next not null value
def carry_on(values, column):
    not_none_indices = numpy.where(values[:,column]!=None)[0]		# finds all the not None values' indices within the column
    for k in range(0, len(not_none_indices)-1):
        for i in range(not_none_indices[k], not_none_indices[k+1]):
            values[i, column] = float(values[not_none_indices[k], column])
    values[not_none_indices[len(not_none_indices)-1]:len(values), column] = float(values[not_none_indices[len(not_none_indices)-1], column]) # complete the column with the last value of sentiment
    
    return values
	
# load a benchmark dataset about pollution in china	
def load_pollution_datasets(pollution_dataset_path='', pollution_dataset_file = '', prepare_pollution_dataset = False, pollution_dataset_file_raw = '', output_size = 1, verbose = True):
    # Prepare benchmark dataset
    if prepare_pollution_dataset == True:
        benchmark_prepare_data(pollution_dataset_path, pollution_dataset_file_raw, pollution_dataset_file)
		   	
    # load dataset
    dataset = read_csv(pollution_dataset_path+'/'+pollution_dataset_file, header=0, index_col=0)
    values = dataset.values

    # integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])

    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # drop columns we don't want to predict
    n_train_elements = 365 * 24    													#1 year of 5 mins data
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    if verbose == True: 
        print(reframed.head())

    # split into train and test sets
    values = reframed.values

    train = values[:n_train_elements, :]
    test  = values[n_train_elements:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-output_size], train[:, -output_size:]				# extracts target of output_size elements from train set and test set
    test_X,  test_y  = test [:, :-output_size], test[:, -output_size:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X  = test_X.reshape ((test_X.shape[0],  1, test_X.shape [1]))
    if parameters.verbose == True: 
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return [train_X, train_y, test_X, test_y, scaler]

def load_bitcoin_datasets(bitcoin_dataset_path='', bitcoin_dataset_file = '', training_set_n_years = 0, training_set_percentage = 0, dataset_sampling_interval = '5m', dataset_sampling_possible_intervals = '5m', full_dataset = False,  T = 1, output_size = 1, normalize = True, input_columns='', output_columns='', verbose = True, remove_first_column = True, skip_rounding = True):
    # load dataset
    dataset_sampling_possible_intervals_list = dataset_sampling_possible_intervals.split(',')
    if remove_first_column == True: 
        start_col = 1
    else:
        start_col = 0
    dataset = read_csv(bitcoin_dataset_path+'/'+bitcoin_dataset_file, header=0, index_col=start_col)
    values = dataset.values
    if remove_first_column == True: 
        values = values[:,1:]
    input_columns_list = input_columns.split(',')
    input_dim = values.shape[1]
    selected_input_dim = len(input_columns_list)
    
    train, valid, test, test_no_labels, scaler, n_train_elements, n_valid_test_elements = drop_columns_and_split(values, input_dim, selected_input_dim, output_size, input_columns, output_columns, training_set_n_years, training_set_percentage, dataset_sampling_interval, dataset_sampling_possible_intervals_list, full_dataset, T, normalize, verbose, skip_rounding, True)  	
    

    # split into input and outputs
    train_X, train_y = train[:, :-output_size], train[:, -output_size:]				# extracts target of output_size elements from train set and test set
    valid_X, valid_y = valid[:, :-output_size], valid[:, -output_size:]
    test_X, test_y = test[:, :-output_size], test[:, -output_size:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    test_no_labels = test_no_labels.reshape((test_no_labels.shape[0], 1, test_no_labels.shape[1]))

    if verbose == True: 
        print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape, test_X.shape, test_y.shape, test_no_labels.shape)

    return [train_X, train_y, valid_X, valid_y, test_X, test_y, test_no_labels, scaler, input_dim, selected_input_dim]

def load_bitcoin_from_db(db_connection_sequence = None, 
                         crypto_orig_type = "BTC", 
                         crypto_dest_type = "ETH", 
                         training_set_n_years = 0, 
                         training_set_percentage = 0, 
                         dataset_sampling_interval = '5m', 
                         dataset_sampling_possible_intervals = '5m', 
                         full_dataset = False, 
                         T = 1, 
                         output_size = 1, 
                         normalize = True, 
                         input_columns='',
                         n_sentiment_columns = 1,       
                         output_columns='', 
                         export_input_db_into_csv = False, 
                         input_db_into_csv_file_path_name = "", 
                         verbose = True):
    loading_start_time = timeit.default_timer()

    if verbose == True: 
        print('Loading: ', crypto_orig_type + '/' + crypto_dest_type)
    [host, user, password, db, charset] = db_connection_sequence
    Connection_Stored_Exchange = pymysql.connect(host=host, user=user, password=password, db=db, charset=charset)
    Connection_Stored_Exchange_Cursor = Connection_Stored_Exchange.cursor() 
    dataset_sampling_possible_intervals_list = dataset_sampling_possible_intervals.split(',')
    if dataset_sampling_interval not in dataset_sampling_possible_intervals_list:
        print("\nData points interval not supported: " + dataset_sampling_interval)
        sys.exit("")
    if full_dataset == True:
        #market_and_sentiment_input_stored_proc = 'sp_GetPoloniexAndSentiment'                   # full dataset reads always 5 mins
        market_and_sentiment_input_stored_proc = 'sp_GetPoloniexAndSentimentV2'                  # New version with technical indicators
    else:     
        if dataset_sampling_interval == dataset_sampling_possible_intervals_list[0]: 
            market_and_sentiment_input_stored_proc = 'sp_GetPoloniexAndSentiment'
        else: 
            market_and_sentiment_input_stored_proc = 'sp_GetPoloniexAndSentiment' + '_' + dataset_sampling_interval
        
    Connection_Stored_Exchange_Cursor.callproc(market_and_sentiment_input_stored_proc, [crypto_orig_type, crypto_dest_type, 0])     # keep the third argument to 0 to read everything
    if verbose == True: 
        print ('Time: %.2f minutes\n' % ((timeit.default_timer() - loading_start_time) / 60.)) 

    values = []
    dates = []
    lines = []
    for Exchanges in Connection_Stored_Exchange_Cursor:
        lines.append(Exchanges)
        values.append(Exchanges[3:])
        dates.append(Exchanges[2]) 
        
    values = numpy.asarray(values) 
    dates = numpy.asarray(dates) 
    
    if export_input_db_into_csv == True: 
        try: 
            numpy.savetxt(input_db_into_csv_file_path_name, lines, delimiter = ",", fmt = "%s")
        except UnicodeEncodeError: 
            print("\nExport of Input from DB into CSV Error.\n")
			
    input_columns_list = input_columns.split(',')
    input_dim = values.shape[1]
    selected_input_dim = len(input_columns_list)

    #sentiment_columns_list = list(map(int, input_columns_list))[7:input_dim]			        # extract just the columns of the sentiment
    sentiment_columns_list = list(map(int, input_columns_list))[7:7+n_sentiment_columns]		# extract just the columns of the sentiment
    sentiment_input_dim = len(sentiment_columns_list)
    for sentiment_column in sentiment_columns_list: 			# loop over all the sentiment columns
        values = carry_on(values, sentiment_column-1)
    numpy.place(values, values == None, 0.0)			        # replace all None with 0

    # drop columns we don't want to predict
    train, valid, test, test_no_labels, scaler, n_train_elements, n_valid_test_elements = drop_columns_and_split(values, input_dim, selected_input_dim, output_size, input_columns, output_columns, training_set_n_years, training_set_percentage, dataset_sampling_interval, dataset_sampling_possible_intervals_list, full_dataset, T, normalize, verbose, False, True)  	
	
    train_dates = dates[:n_train_elements]
    valid_dates = dates[n_train_elements:n_train_elements+n_valid_test_elements]
    test_dates  = dates[n_train_elements+n_valid_test_elements:]
    
    # split into input and outputs
    train_X, train_y = train[:, :-output_size], train[:, -output_size:]				# extracts target of output_size elements from train set and test set
    valid_X, valid_y = valid[:, :-output_size], valid[:, -output_size:]
    test_X, test_y = test[:, :-output_size], test[:, -output_size:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
    test_X  = test_X.reshape ((test_X.shape [0], 1, test_X.shape [1]))
    test_no_labels = test_no_labels.reshape((test_no_labels.shape[0], 1, test_no_labels.shape[1]))

    if verbose == True: 
        print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape, test_X.shape, test_y.shape, test_no_labels.shape)

    return [train_X, train_y, valid_X, valid_y, test_X, test_y, test_no_labels, scaler, train_dates, test_dates, input_dim, selected_input_dim]
#-----------------------------------
#---------- Questa funzione serve  |
#-----------------------------------
def load_bitcoin_from_db_for_prediction(db_connection_sequence = None, 
                                        crypto_orig_type = "BTC", 
                                        crypto_dest_type = "ETH", 
                                        prediction_data_size = 10, 
                                        dataset_sampling_interval = '5m', 
                                        dataset_sampling_possible_intervals = '5m', 
                                        full_dataset = False, 
                                        T = 1, 
                                        output_size = 1, 
                                        normalize = True, 
                                        input_columns='', 
                                        output_columns='', 
                                        export_input_db_into_csv = False, 
                                        input_db_into_csv_file_path_name = "", 
                                        verbose = True):
    loading_start_time = timeit.default_timer()

    if verbose == True: 
        print('Loading: ', crypto_orig_type+'/'+crypto_dest_type)
    [host, user, password, db, charset] = db_connection_sequence
    Connection_Stored_Exchange = pymysql.connect(host=host, user=user, password=password, db=db, charset=charset)
    Connection_Stored_Exchange_Cursor = Connection_Stored_Exchange.cursor() 
    # dataset_sampling_possible_intervals_list = dataset_sampling_possible_intervals.split(',')
    # if dataset_sampling_interval not in dataset_sampling_possible_intervals_list:
        # print("\nData points interval not supported: " + dataset_sampling_interval)
        # sys.exit("")
    # if full_dataset == True:
    market_and_sentiment_input_stored_proc = 'sp_GetPoloniexAndSentiment'    # full dataset reads always 5 mins
    # else:     
        # if dataset_sampling_interval == dataset_sampling_possible_intervals_list[0]: 
            # market_and_sentiment_input_stored_proc = 'sp_GetPoloniexAndSentiment'
        # else: 
            # market_and_sentiment_input_stored_proc = 'sp_GetPoloniexAndSentiment' + '_' + dataset_sampling_interval
        
    Connection_Stored_Exchange_Cursor.callproc(market_and_sentiment_input_stored_proc, [crypto_orig_type, crypto_dest_type, prediction_data_size])     
	
    if verbose == True: 
        print ('Time: %.2f minutes\n' % ((timeit.default_timer() - loading_start_time) / 60.)) 

    values = []
    dates  = []
    lines  = []
    for Exchanges in Connection_Stored_Exchange_Cursor:
        lines.append (Exchanges)
        values.append(Exchanges[3:18])
        dates.append (Exchanges[2]) 

    values = numpy.asarray(values) 
    dates  = numpy.asarray(dates) 
    
    if export_input_db_into_csv == True: 
        try: 
            numpy.savetxt(input_db_into_csv_file_path_name, lines, delimiter = ",", fmt = "%s")
        except UnicodeEncodeError: 
            print("\nExport of Input from DB into CSV Error.\n")
			
    input_columns_list = input_columns.split(',')
    input_dim = values.shape[1]
    selected_input_dim = len(input_columns_list)

    sentiment_columns_list = list(map(int, input_columns_list))[7:input_dim]			# extract just the columns of the sentiment
    sentiment_input_dim = len(sentiment_columns_list)
    for sentiment_column in sentiment_columns_list: 			# loop over all the sentiment columns
        values = carry_on(values, sentiment_column-1)
    numpy.place(values, values == None, 0.0)			        # replace all None with 0

    # drop columns we don't want to predict
    test, test_no_labels, scaler = drop_columns_and_split(values, 
	                                                      input_dim, 
	                                                      selected_input_dim, 
	                                                      output_size, 
	                                                      input_columns, 
	                                                      output_columns, 
	                                                      None, 
	                                                      None, 
	                                                      dataset_sampling_interval, 
	                                                      dataset_sampling_possible_intervals_list, 
	                                                      full_dataset, 
	                                                      T, 
	                                                      normalize, 
	                                                      verbose, 
	                                                      False, 
	                                                      False)  	
	
    test_dates = dates
    
    # split into input and outputs
    test_X, test_y = test[:, :-output_size], test[:, -output_size:]

    # reshape input to be 3D [samples, timesteps, features]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    test_no_labels = test_no_labels.reshape((test_no_labels.shape[0], 1, test_no_labels.shape[1]))

    if verbose == True: 
        print(test_X.shape, test_y.shape, test_no_labels.shape)

    return [test_X, test_y, test_no_labels, scaler, test_dates, input_dim, selected_input_dim]
#-----------------------------------
#---------- Questa funzione serve  |
#-----------------------------------
def save_bitcoin_into_db(db_connection_sequence = None, 
                         crypto_orig_type = 'BTC', 
                         crypto_dest_type = 'ETH', 
                         test_dates = None, 
                         sampling_basic_unit = 1, 
                         n_residual_elements = 1, 
                         inv_yhat = None, 
                         dataset_sampling_interval = '5m', 
                         dataset_sampling_possible_intervals = '5m', 
                         full_dataset = False, 
                         verbose = True): 
    if verbose == True: 
        print('Saving Predictions onto MySql DB...')
    [host, user, password, db, charset] = db_connection_sequence
    Connection_Stored_SetPredictions = pymysql.connect(host=host, user=user, password=password, db=db, charset=charset)
    Connection_Stored_SetPredictions_Cursor = Connection_Stored_SetPredictions.cursor() 
    # dataset_sampling_possible_intervals_list = dataset_sampling_possible_intervals.split(',')

    test_dates = test_dates + timedelta(minutes=sampling_basic_unit*n_residual_elements)  # increments dates for the prediction time
    
    # Opening the database
    if full_dataset == True:
        market_and_sentiment_prediction_stored_proc = 'sp_SetPoloniexPrediction_Full'   # full dataset reads always 5 mins
    # else:     
        # if dataset_sampling_interval == dataset_sampling_possible_intervals_list[0]: 
            # market_and_sentiment_prediction_stored_proc = 'sp_SetPoloniexPrediction'
        # else: 
            # market_and_sentiment_prediction_stored_proc = 'sp_SetPoloniexPrediction' + '_' + dataset_sampling_interval
    # # Writing
    # if full_dataset == True:
        for test_date, single_yhat in zip(test_dates, inv_yhat):
            sp_arguments = [dataset_sampling_interval, crypto_orig_type, crypto_dest_type, test_date.strftime('%Y-%m-%d %H:%M:%S'), float("%.8f" % single_yhat), 0.0]
            Connection_Stored_SetPredictions_Cursor.callproc(market_and_sentiment_prediction_stored_proc, sp_arguments)
    # else:     
        # for test_date, single_yhat in zip(test_dates, inv_yhat):
            # sp_arguments = [crypto_orig_type, crypto_dest_type, test_date.strftime('%Y-%m-%d %H:%M:%S'), float("%.8f" % single_yhat), 0.0]
            # Connection_Stored_SetPredictions_Cursor.callproc(market_and_sentiment_prediction_stored_proc, sp_arguments)

    Connection_Stored_SetPredictions.commit()
    Connection_Stored_SetPredictions_Cursor.close()
    del Connection_Stored_SetPredictions_Cursor
    del Connection_Stored_SetPredictions
