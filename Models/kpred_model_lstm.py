'''
Created on 27/04/2017

@author: Francesco Pugliese, Matteo Testi
'''
#from keras.layers import LSTM, CuDNNLSTM, Dense
from keras.layers import LSTM, Dense
from keras.models import Sequential
import pdb

class PredModelLSTM:
    @staticmethod
    def build(input_length, vector_dim, output_size, batch_size, n_hiddens, summary, stateful):
      
        # initialize the model
        deepnetwork = Sequential()
        #deepnetwork.add(CuDNNLSTM(n_hiddens, input_shape = (input_length, vector_dim)))   
        if stateful == True: 
            deepnetwork.add(LSTM(n_hiddens, batch_input_shape = (batch_size, 1, vector_dim), stateful = stateful))   
        else: 
            deepnetwork.add(LSTM(n_hiddens, input_shape = (input_length, vector_dim), stateful = stateful))   
        
        #deepnetwork.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences = True, input_shape = (input_length, vector_dim)))   # return_sequences to stack more LSTM
        deepnetwork.add(Dense(output_size))
		
        if summary==True:
            deepnetwork.summary()

        return deepnetwork

