'''
Created on 08/01/2018

@author: Francesco Pugliese
'''

# Preprocessing
from Preprocessing.preprocessing import load_pollution_datasets, load_bitcoin_datasets, load_bitcoin_from_db
from Settings.settings import SetParameters

# Models imports
from Models.kpred_model_sae import PredModelSAE
from Models.kpred_model_lstm import PredModelLSTM
from Models.kpred_model_gru import PredModelGRU 
from Models.kpred_model_conv_lstm import PredModelConvLSTM
from Models.kpred_model_rec_conv_net import PredModelRecConvNet

# Miscellaneous
from Misc.utils import set_cpu_or_gpu

# Visualization
from View.view import plot_dataset, plot_loss, print_computation_time_in_days

# Keras imports
from keras.callbacks import ModelCheckpoint, CSVLogger

# Other imports
import timeit
from pandas import read_csv
import pdb
from datetime import datetime

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
        
for si in range(total_dataset_sampling_loop_intervals): 
    training_start_time = timeit.default_timer()
  
    for cd in range(total_crypto_dest): 
    
        if parameters.loop_on_crypto_dest_type == True and parameters.input_from_db == True: 
            crypto_dest_type = crypto_dest_type_possible_intervals_list[cd] 
    
        if parameters.loop_on_dataset_sampling_intervals == True: 
            dataset_sampling_interval = dataset_sampling_loop_intervals_list[si] 

        if parameters.verbose == False: 
            print('Training on: ', parameters.crypto_orig_type+'/'+crypto_dest_type)

        files_prefix = parameters.crypto_orig_type+"-"+crypto_dest_type+"_T"+str(parameters.T)+"_"+dataset_sampling_interval+"_"+parameters.neural_model+"_"		# default prefix of each saved file
        if parameters.full_dataset == True:
            files_prefix = files_prefix + "full_" 
        if parameters.stateful == True:
            files_prefix = files_prefix + "stateful_" 
        
        if parameters.pollution_benchmark == True:
            train_X, train_y, test_X, test_y, scaler = load_pollution_datasets(parameters.pollution_dataset_path, parameters.pollution_dataset_file, parameters.prepare_pollution_dataset, parameters.pollution_dataset_file_raw, parameters.output_size, parameters.verbose)
        elif parameters.bitcoin == True: 
            if parameters.input_from_db == True:
                train_X, train_y, valid_X, valid_y, test_X, test_y, test_no_labels, scaler, train_dates, test_dates, input_dim, selected_input_dim = load_bitcoin_from_db(parameters.db_connection_sequence, parameters.crypto_orig_type, crypto_dest_type, parameters.training_set_n_years, parameters.training_set_percentage, dataset_sampling_interval, parameters.dataset_sampling_possible_intervals, parameters.full_dataset, parameters.T, parameters.output_size, parameters.normalize, parameters.input_columns, parameters.n_sentiment_columns, parameters.output_columns, parameters.export_input_db_into_csv, parameters.output_path+'/'+files_prefix+parameters.input_db_into_csv_file, parameters.verbose)
            else:
                # do not include sentiment with load from csv datasets, only from db
                train_X, train_y, valid_X, valid_y, test_X, test_y, test_no_labels, scaler, input_dim, selected_input_dim = load_bitcoin_datasets(parameters.bitcoin_dataset_path, parameters.bitcoin_dataset_file, parameters.training_set_n_years, parameters.training_set_percentage, dataset_sampling_interval, parameters.dataset_sampling_possible_intervals, parameters.full_dataset, parameters.T, parameters.output_size, parameters.normalize, parameters.input_columns, parameters.output_columns, parameters.verbose, parameters.remove_first_column, parameters.skip_rounding)

        if parameters.neural_model == 'sae_lstm':
            models, sae_model = PredModelSAE.build(input_length=train_X.shape[1], vector_dim=train_X.shape[2], output_size = parameters.output_size, depth = parameters.sae_depth, encoding_dim = parameters.sae_encoding_dim, summary=True)	

            sae_checkPoint=ModelCheckpoint(parameters.models_path+'/'+files_prefix+"sae_part_"+parameters.model_file, save_weights_only=True, monitor='val_loss', verbose=verbose_model_check_point, save_best_only=True, mode='min')

            train = train_X.reshape((train_X.shape[0], train_X.shape[2]))
            test = test_X.reshape((test_X.shape[0], test_X.shape[2]))
            valid = valid_X.reshape((valid_X.shape[0], valid_X.shape[2]))

            l = 0
            for m in models:
                l = l + 1
                print('\n\nTraining Stacked Autoencoder layer: ', l)
                #print('\n')
                m.compile(loss='mae', optimizer='adam')
                if l<parameters.sae_depth:
                    m.fit(train, train, epochs=parameters.sae_epochs_number, batch_size=parameters.sae_batch_size, verbose=verbose_fit, shuffle=False, validation_data=(test, test))
                else:
                    # save best model with the last greedy layer
                    m.fit(train, train, epochs=parameters.sae_epochs_number, callbacks = [sae_checkPoint], batch_size=parameters.sae_batch_size, verbose=verbose_fit, shuffle=False, validation_data=(test, test))
                
            # Load the saved best stacked autoencoder
            if os.path.isfile(parameters.models_path+'/'+files_prefix+"sae_part_"+parameters.model_file):
                models[parameters.sae_depth - 1].load_weights(parameters.models_path+'/'+files_prefix+"sae_part_"+parameters.model_file) 
            else:
                print('\nPre-trained stacked autoencoder not found: %s.' % (parameters.models_path+'/'+files_prefix+"sae_part_"+parameters.model_file))
                sys.exit("")

            train_X = sae_model.predict(train)
            test_X = sae_model.predict(test)
            valid_X = sae_model.predict(valid)
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            valid_X = valid_X.reshape((valid.shape[0], 1, valid_X.shape[1]))
        
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
            deepnetwork = PredModelLSTM.build(input_length=train_X.shape[1], vector_dim=train_X.shape[2], output_size = parameters.output_size, batch_size = parameters.batch_size, n_hiddens = parameters.n_hiddens, summary=summary, stateful = parameters.stateful)	
        
        if parameters.neural_model == 'gru':
            deepnetwork = PredModelGRU.build(input_length=train_X.shape[1], vector_dim=train_X.shape[2], output_size = parameters.output_size, batch_size = parameters.batch_size, n_hiddens = parameters.n_hiddens, summary=summary, stateful = parameters.stateful)	

        if parameters.neural_model == 'conv_lstm':
            deepnetwork = PredModelConvLSTM.build(input_length=train_X.shape[1], vector_dim=train_X.shape[2], output_size = parameters.output_size, batch_size = parameters.batch_size, n_hiddens = parameters.n_hiddens, summary=summary, stateful = parameters.stateful)	

        if parameters.neural_model == 'rec_conv_net':
            [deepnetwork, deepnetwork_with_best_attributes] = PredModelRecConvNet.build(input_length=train_X.shape[1], vector_dim=train_X.shape[2], output_size = parameters.output_size, batch_size = parameters.batch_size, n_hiddens = parameters.n_hiddens, summary=summary, stateful = parameters.stateful)	

        deepnetwork.compile(loss='mae', optimizer='adam')

        default_callbacks = []

        if parameters.save_log == True:
            csvLogger = CSVLogger(parameters.log_path+'/'+files_prefix+parameters.log_file)
            default_callbacks = default_callbacks+[csvLogger]

        if parameters.save_best_model == True:
            checkPoint=ModelCheckpoint(parameters.models_path+'/'+files_prefix+parameters.model_file, save_weights_only=True, monitor='val_loss', verbose=verbose_model_check_point, save_best_only=True, mode='min')
            default_callbacks = default_callbacks+[checkPoint]

        # fit network
        history = deepnetwork.fit(train_X, train_y, epochs=parameters.epochs_number, callbacks = default_callbacks, batch_size=parameters.batch_size, validation_data=(valid_X, valid_y), verbose=verbose_fit, shuffle=False)

        if parameters.save_best_model == False:                                    
            deepnetwork.save_weights(parameters.models_path+'/'+files_prefix+parameters.model_file)

    training_end_time = timeit.default_timer()
    
    if parameters.verbose == True: 
        print ('\n\n')
    print ('%s Training time: %.2f minutes\n' % (crypto_dest_type, (training_end_time - training_start_time) / 60.)) 
    
global_end_time = timeit.default_timer()

global_time_in_sec = global_end_time - global_start_time
print ('\n\n')
print_computation_time_in_days(global_time_in_sec)                  # displays computation time in days, hours, etc
print ('\n\n')

