'''
Created on 30/12/2018

@author: Alberto Piva, Francesco Pugliese
'''
from keras.layers import Dense, Input
from keras.models import Model
import pdb

class PredModelSAE:
    @staticmethod
    def build(input_length, vector_dim, output_size, depth, encoding_dim, summary):
        input_series = Input(shape=(vector_dim,))

        hiddens = []
        hidden0 = Dense(encoding_dim, activation='relu')(input_series)
        hiddens.append(hidden0)
        for l in range(1, depth): 
            new_hidden = Dense(encoding_dim, activation='relu')(hiddens[l-1])  # Build the stack of hiddens, every hidden depends from the previous hidden 
            hiddens.append(new_hidden)
            
        decoders = []
        decoder0 = Dense(vector_dim, activation='sigmoid')(hidden0)
        decoders.append(decoder0)
        for l in range(1, depth): 
            new_decoder = Dense(vector_dim, activation='sigmoid')(hiddens[l])  # Build the stack of decoders, every decoder depends from the related hidden
            decoders.append(new_decoder)

        models = []    
        for l in range(0, depth): 
            new_model = Model(input_series, decoders[l])                       # List of all the trainable models
            models.append(new_model)
            
        sae_model = Model(input_series, hiddens[depth-1])                      # Whole Stacked Autoencoder Model
            		
        if summary==True:
            models[depth-1].summary()

        return [models, sae_model]

