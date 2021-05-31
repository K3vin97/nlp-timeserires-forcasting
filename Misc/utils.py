import numpy as np
import argparse
import pdb
import os

def set_cpu_or_gpu(parameters):
    # Set CPU or GPU type
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    if parameters.gpu == False: 
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"] = parameters.gpu_id

def parse_arguments(default_config_file):
    # Constructs the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", help="configuration file name", required = False)
    (arg1) = parser.parse_args()
    config_file = arg1.conf
    if config_file is None: 
        config_file = default_config_file              # Default Configuration File
    
    return config_file

# Converts time from seconds to days, hours, mins, secs, millisecs format
def time_from_secs_to_days(time_in_sec):
    secs, msecs = divmod(time_in_sec, 1)
    secs = int(secs) 
    msecs = int(msecs * 1000) 
    mins = secs // 60
    secs = secs % 60
    hours = mins // 60
    mins = mins % 60
    days = hours // 24
    hours = hours % 24
    
    return [days, hours, mins, secs, msecs]
    
# Selector of the numnber of train elements
def n_train_elements_selector(values, training_set_n_years, training_set_percentage, dataset_sampling_interval, dataset_sampling_possible_intervals_list, skip_rounding):
    if training_set_n_years is not None: 
        # drop columns we don't want to predict
        if dataset_sampling_interval == dataset_sampling_possible_intervals_list[0]:
            n_train_elements = 365 * 24 * 12 * training_set_n_years   						    #n years of 5 mins data
            print("\nData points every 5 minutes\n")
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[1]:
            n_train_elements = 365 * 12 * training_set_n_years   	    					    #n years of 2 hours data
            print("\nData points every 2 hours\n")
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[2]:
            n_train_elements = 365 * 6 * training_set_n_years   	    					    #n years of 4 hours data
            print("\nData points every 4 hours\n")
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[3]:
            n_train_elements = 365 * 24 * 4 * training_set_n_years   						    #n years of 15 mins data
            print("\nData points every 15 minutes\n")
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[4]:
            n_train_elements = 365 * training_set_n_years                					    #n years of 24 hours data
            print("\nData points every 24 hours\n")
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[5]:
            n_train_elements = 365 * 24 * 2 * training_set_n_years   						    #n years of 30 mins data
            print("\nData points every 30 minutes\n")
        else: 
            print("\nData points interval not supported: "+dataset_sampling_interval)
            sys.exit("")
    elif training_set_percentage is not None: 
        # drop columns we don't want to predict
        n_elements = int(len(values) * training_set_percentage / 100.0)
        if skip_rounding == True: 
            n_train_elements = n_elements
        else: 
            if dataset_sampling_interval == dataset_sampling_possible_intervals_list[0]:
                n_train_elements = int(n_elements / (24 * 12)) * 24 * 12      			    	    #n groups of 5 mins data, round on days
                print("\nData points every 5 minutes\n")
            elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[1]:
                n_train_elements = int(n_elements / 12) * 12   	             					    #n groups of 2 hours data, round on days
                print("\nData points every 2 hours\n")
            elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[2]:
                n_train_elements = int(n_elements / 6) * 6          	    					    #n groups of 4 hours data, round on days
                print("\nData points every 4 hours\n")
            elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[3]:
                n_train_elements = int(n_elements / (24 * 4)) * 24 * 4                              #n groups of 15 mins data, round on days
                print("\nData points every 15 minutes\n")
            elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[4]:
                n_train_elements = n_elements                                                       #n groups of 24 hours data, round on days
                print("\nData points every 24 hours\n")
            elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[5]:
                n_train_elements = int(n_elements / (24 * 2)) * 24 * 2   						    #n groups of 30 mins data, round on days
                print("\nData points every 30 minutes\n")
            else: 
                print("\nData points interval not supported: "+dataset_sampling_interval)
                sys.exit("")
    else: 
        print("\nTrain Set split percentage parameter error.")
        sys.exit("")

    return n_train_elements
    
    
# Selector of number of outputs
def n_out_selector(full_dataset, dataset_sampling_interval, dataset_sampling_possible_intervals_list):
    if full_dataset == True:
        if dataset_sampling_interval == dataset_sampling_possible_intervals_list[0]:
            n_out = 1   						        # 5 mins data
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[1]:
            n_out =  24   						        # 2 hours data
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[2]:
            n_out = 48     						        # 4 hours data
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[3]:
            n_out = 3        						    # 15 mins data
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[4]:
            n_out = 288      						    # 24 hours data
        elif dataset_sampling_interval == dataset_sampling_possible_intervals_list[5]:
            n_out = 6          						    # 30 mins data
        else: 
            print("\nData points interval not supported: "+dataset_sampling_interval)
            sys.exit("")
    else: 
        n_out = 1

    return n_out