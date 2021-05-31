'''
Created on 27/04/2017

@author: Francesco Pugliese, Matteo Testi
'''
from keras.layers import GRU, Dense
from keras.models import Sequential
import pdb

class PredModelGRU:
    @staticmethod
    def build(input_length, vector_dim, output_size, batch_size, n_hiddens, summary, stateful):
      
        # initialize the model
        deepnetwork = Sequential()
        if stateful == True: 
            deepnetwork.add(GRU(n_hiddens, batch_input_shape = (batch_size, 1, vector_dim), stateful = stateful))   
        else: 
            deepnetwork.add(GRU(n_hiddens, input_shape = (input_length, vector_dim), stateful = stateful))   

        deepnetwork.add(Dense(output_size))
		
        if summary==True:
            deepnetwork.summary()

        return deepnetwork

