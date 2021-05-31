'''
Created on 11/01/2018

@author: Francesco Pugliese
'''

# Miscellaneous
from Misc.utils import time_from_secs_to_days

from pandas import read_csv
from matplotlib import pyplot
import pdb

# Plot inputs
def plot_dataset(pred_dataset_path='', pred_dataset_file=''):
    # load dataset
    dataset = read_csv(pred_dataset_path+'/'+pred_dataset_file, header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
	    pyplot.subplot(len(groups), 1, i)
	    pyplot.plot(values[:, group])
	    pyplot.title(dataset.columns[group], y=0.5, loc='right')
	    i += 1
    pyplot.show(block=False)
    pyplot.pause(1)
	
# Plot the loss function chart
def plot_loss(history = None):
    pyplot.figure()												# generate a new window
    pyplot.plot(history['loss'], label='train')
    pyplot.plot(history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show(block=False)
    pyplot.pause(1)

# Plot predictions and real y chart
def plot_predictions(predictions_view_path_file, rmse, mape, inv_yhat = None, inv_y = None, view = False, pause_time = 1):
    pyplot.figure()												# generate a new window
    pyplot.plot(inv_yhat, label='Predicted Y', color = 'blue')
    pyplot.plot(inv_y, label='Real Y', color = 'green')
    pyplot.legend()
    pyplot.title(predictions_view_path_file+"\nRMSE: "+str(rmse)+'\nMAPE: '+str(mape))
    pyplot.savefig(predictions_view_path_file, dpi = 600)
    if view == True: 
        pyplot.show(block=False)
        pyplot.pause(pause_time)

def print_computation_time_in_days(time_in_sec):
    days, hours, mins, secs, msecs = time_from_secs_to_days(time_in_sec)
    
    print ('Global computation time:')     
    print ('Days: %i' % (days))     
    print ('Hours : %i ' % (hours))     
    print ('Minutes: %i' % (mins))     
    print ('Seconds : %i ' % (secs))     
    print ('Milli seconds : %i ' % (msecs))     
