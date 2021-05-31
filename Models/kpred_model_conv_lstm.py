'''
Created on 27/04/2017
Modified on 12/07/2017


@author: Francesco Pugliese
'''
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from keras.models import Sequential
import pdb

class PredModelConvLSTM:
    @staticmethod
    def build(input_length, vector_dim, output_size, batch_size, n_hiddens, summary, stateful):
      
        # initialize the model
        deepnetwork = Sequential()
        
        deepnetwork.add(Conv1D(filters=128, kernel_size=5, activation = 'relu', padding = 'same', input_shape = (input_length, vector_dim)))
        #deepnetwork.add(MaxPooling1D(5))
        deepnetwork.add(LSTM(n_hiddens, dropout=0.2, recurrent_dropout=0.2))
        deepnetwork.add(Dense(output_size))

        if summary==True:
            deepnetwork.summary()

        return deepnetwork

