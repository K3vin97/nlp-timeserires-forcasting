import pdb

# make a single prediction considering normalized data
# usage es: yp = single_prediction(numpy.asarray[50.0,0.00900005,50.0,0.00900005,10.65496054,1033.26076055,0.01031197,0.5],15,8,deepnetwork)
def single_prediction(single_test_X, input_dim, selected_input_dim, deepnetwork): 
    # ensure all data is float
    values = single_test_X.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    yhat = deepnetwork.predict(test_X)                  # prediction of y values
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((test_X[:, 0:(int(output_columns)-1)], yhat, test_X[:, int(output_columns):selected_input_dim+1]), axis=1)
    inv_yhat = numpy.hstack((inv_yhat, numpy.zeros((len(inv_yhat),input_dim-selected_input_dim))))  	    # add columns of 0 up to the end of the array
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,(int(output_columns)-1)]
    
    return inv_yhat
