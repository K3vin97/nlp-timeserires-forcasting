'''
Created on 23/03/2018


@author: Francesco Pugliese
'''
# Miscellaneous
from Misc.utils import parse_arguments

import configparser
import argparse
import platform

class SetParameters:
    
    def __init__(self, conf_file_path, conf_file_name, OS):
        # Class Initialization (constructor) 
        self.conf_file_path = conf_file_path
        self.conf_file_name = conf_file_name
        
        # System
        self.gpu = False
        self.gpu_id = '0'
        
        # Preprocessing
        self.T = 1
        self.normalize = True

        # Dataset
        self.bitcoin = False
        self.crypto_orig_type = 'BCN'
        self.crypto_dest_type = 'ETH'
        self.crypto_orig_type_possible_intervals = 'BTC'    
        self.crypto_dest_type_possible_intervals = 'ETH'        
        self.input_from_db = False
        self.training_set_n_years = 1
        self.training_set_percentage = 80
        self.dataset_sample_interval = '5m'
        self.dataset_sampling_loop_intervals = '5m'
        self.bitcoin_dataset_path = '' 
        self.bitcoin_dataset_file = ''
        self.remove_first_column = False
        self.skip_rounding = False
        self.full_dataset = False
        self.loop_on_crypto_orig_type = False
        self.loop_on_crypto_dest_type = False
        self.loop_on_dataset_sampling_intervals = False
        
        # Prediction
        self.prediction_data_metrics_evaluation = True
        self.prediction_data_size = 10

        # MySql
        self.db_connection_sequence = None

        # Input
        self.input_columns = ''
        self.n_sentiment_columns = 0
        
		# Output
        self.output_columns = ''
        self.save_log = False
        self.log_path = '../Log'
        self.log_file = 'training.log'
        self.export_predictions_into_db = False 
        self.export_predictions_into_csv = False
        self.export_metrics_into_csv = False
        self.export_input_db_into_csv = False
        self.input_db_into_csv_file = 'input_from_db.csv'
        self.output_path = '../Output'
        self.output_file = 'predictions.csv'
        self.metrics_file = 'metrics.csv'
		
        # Visualization
        self.dataset_view = True
        self.history_view = True
        self.predictions_view = True
        self.predictions_view_file = 'input_from_db.csv'
        self.press_key = True
        self.verbose = True

        # Model 
        self.neural_model = 'lstm'
        self.n_hiddens = 5
        self.models_path = '../SavedModels'
        self.model_file = "best_prediction_deep_model.snn"
        self.sae_depth = 4                                 																																																	
        self.sae_encoding_dim = 4           
        self.stateful = True

		# Training
        self.epochs_number = 20
        self.batch_size = 32
        self.save_best_model = False
        self.sae_batch_size = 72
        self.sae_epochs_number = 300


        # Topology
        self.output_size = 1

		# Benchmarking
        self.pollution_benchmark = False
        self.pollution_dataset_pathLIN ='../../../Market_Data'
        self.pollution_dataset_pathWIN = 'G:/Market Data'
        self.pollution_dataset_file_raw = 'PRSA_data_2010.1.1-2014.12.31.csv'
        self.pollution_dataset_file = 'pollution.csv'
        self.pollution_epochs_number = 18
        self.pollution_output_size = 1
        self.pollution_batch_size = 72                                  																																																	
        self.prepare_pollution_dataset = False

		# Others
        self.OS = OS
        
        # Global Constants
        self.dataset_sampling_possible_intervals = "5m,02h,04h,15m,24h,30m" 
        self.sampling_basic_unit = 5
	
    def read_config_file(self):
        # Read the Configuration File
        config = configparser.ConfigParser()
        config.read(self.conf_file_path+'/'+self.conf_file_name)
        config.sections()

        # System
        self.gpu = config.getboolean('System', 'gpu')
        self.gpu_id = config.get('System','gpu_id')

        # Preprocessing
        self.T = config.getint('Preprocessing', 'T')						# depth in time of the RNN
        self.normalize = config.getboolean('Preprocessing', 'normalize')				

        # Dataset
        self.bitcoin = config.getboolean('Dataset', 'bitcoin')
        self.crypto_orig_type = config.get('Dataset', 'crypto_orig_type')
        self.crypto_orig_type = self.crypto_orig_type.upper()
        self.crypto_dest_type = config.get('Dataset', 'crypto_dest_type')
        self.crypto_dest_type = self.crypto_dest_type.upper()
        self.crypto_orig_type_possible_intervals = config.get('Dataset', 'crypto_orig_type_possible_intervals')
        self.crypto_orig_type_possible_intervals = self.crypto_orig_type_possible_intervals.upper()
        self.crypto_dest_type_possible_intervals = config.get('Dataset', 'crypto_dest_type_possible_intervals')
        self.crypto_dest_type_possible_intervals = self.crypto_dest_type_possible_intervals.upper()
        self.input_from_db = config.getboolean('Dataset', 'input_from_db')
        self.training_set_n_years = config.get('Dataset', 'training_set_n_years')
        try: 
            self.training_set_n_years = int(self.training_set_n_years)
        except ValueError: 
            self.training_set_n_years = None
        self.training_set_percentage = config.get('Dataset', 'training_set_percentage')
        try: 
            self.training_set_percentage = int(self.training_set_percentage)
        except ValueError: 
            self.training_set_percentage = None
        self.dataset_sampling_interval = config.get('Dataset', 'dataset_sampling_interval')
        self.dataset_sampling_loop_intervals = config.get('Dataset', 'dataset_sampling_loop_intervals')
        if self.bitcoin == True:
            if self.OS == "Linux":
                self.bitcoin_dataset_path = config.get('Dataset', 'bitcoin_dataset_pathLIN')
            elif self.OS == "Windows": 
                self.bitcoin_dataset_path = config.get('Dataset', 'bitcoin_dataset_pathWIN')
        self.bitcoin_dataset_file = config.get('Dataset', 'bitcoin_dataset_file')
        self.remove_first_column = config.getboolean('Dataset', 'remove_first_column')
        self.skip_rounding = config.getboolean('Dataset', 'skip_rounding')
        self.full_dataset = config.getboolean('Dataset', 'full_dataset')
        self.loop_on_crypto_orig_type = config.getboolean('Dataset', 'loop_on_crypto_orig_type')
        self.loop_on_crypto_dest_type = config.getboolean('Dataset', 'loop_on_crypto_dest_type')
        self.loop_on_dataset_sampling_intervals = config.getboolean('Dataset', 'loop_on_dataset_sampling_intervals')

        # Prediction
        self.prediction_data_metrics_evaluation = config.getboolean('Prediction', 'prediction_data_metrics_evaluation')
        self.prediction_data_size = config.get('Prediction', 'prediction_data_size')
        try: 
            self.prediction_data_size = int(self.prediction_data_size)
        except ValueError: 
            self.prediction_data_size = None

        # MySql Database	
        if self.input_from_db == True: 
            host = config.get('MySql', 'host')
            user = config.get('MySql', 'user')
            password = config.get('MySql', 'password')
            db = config.get('MySql', 'db')
            charset = config.get('MySql', 'charset')
            self.db_connection_sequence = [host, user, password, db, charset] 
		
        # Input
        self.input_columns = config.get('Input', 'input_columns')
        self.n_sentiment_columns = config.getint('Input', 'n_sentiment_columns')
        
		# Output
        self.output_columns = config.get('Output', 'output_columns')
        self.save_log = config.getboolean('Output', 'save_log')
        self.log_path = config.get('Output', 'log_path')
        self.log_file = config.get('Output', 'log_file')
        self.export_predictions_into_db = config.getboolean('Output', 'export_predictions_into_db')
        self.export_predictions_into_csv = config.getboolean('Output', 'export_predictions_into_csv')
        self.export_metrics_into_csv = config.getboolean('Output', 'export_metrics_into_csv')
        self.export_input_db_into_csv = config.getboolean('Output', 'export_input_db_into_csv')
        self.input_db_into_csv_file = config.get('Output', 'input_db_into_csv_file')
        self.output_path = config.get('Output', 'output_path')
        self.output_file = config.get('Output', 'output_file')
        self.metrics_file = config.get('Output', 'metrics_file')
		
		# Visualization
        self.dataset_view = config.getboolean('Visualization', 'dataset_view')
        self.history_view = config.getboolean('Visualization', 'history_view')
        self.predictions_view = config.getboolean('Visualization', 'predictions_view')
        self.predictions_view_file = config.get('Visualization', 'predictions_view_file')
        self.press_key = config.getboolean('Visualization', 'press_key')
        self.verbose = config.getboolean('Visualization', 'verbose')
		
        # Model 
        self.neural_model = config.get('Model', 'neural_model')
        self.n_hiddens = config.getint('Model', 'n_hiddens')
        self.models_path = config.get('Model', 'models_path')
        self.model_file = config.get('Model', 'model_file')
        self.sae_depth = config.getint('Model', 'sae_depth')                                 																																																	
        self.sae_encoding_dim = config.getint('Model', 'sae_encoding_dim')                                 																																																	
        self.stateful = config.getboolean('Model', 'stateful')
        
		# Training
        self.epochs_number = config.getint('Training', 'epochs_number')
        self.batch_size = config.getint('Training', 'batch_size')
        self.save_best_model = config.getboolean('Training', 'save_best_model')
        self.sae_batch_size = config.getint('Training', 'sae_batch_size')
        self.sae_epochs_number = config.getint('Training', 'sae_epochs_number')
		
        # Topology
        self.output_size = config.getint('Topology', 'output_size')

		# Benchmarking
        self.pollution_benchmark = config.getboolean('Benchmarking', 'pollution_benchmark')
        if self.pollution_benchmark == True:
            if self.OS == "Linux":
                self.pollution_dataset_path = config.get('Benchmarking', 'pollution_dataset_pathLIN')
            elif self.OS == "Windows": 
                self.pollution_dataset_path = config.get('Benchmarking', 'pollution_dataset_pathWIN')
            self.pollution_dataset_file_raw = config.get('Benchmarking', 'pollution_dataset_file_raw')
            self.pollution_dataset_file = config.get('Benchmarking', 'pollution_dataset_file')
            self.prepare_pollution_dataset = config.getboolean('Benchmarking', 'prepare_pollution_dataset')
            self.epochs_number = config.getint('Benchmarking', 'pollution_epochs_number')
            self.output_size = config.getint('Benchmarking', 'pollution_output_size')
            self.number_of_batches = config.getint('Benchmarking', 'pollution_number_of_batches')
		
        return self		
        
    @staticmethod
    def init_by_config_file():
        # Operating System
        OS = platform.system()						# returns 'Windows', 'Linux', etc

        # Read the Configuration File
        config_file = parse_arguments("enginepred.ini")
        parameters = SetParameters("../Conf", config_file, OS).read_config_file()

        return parameters

