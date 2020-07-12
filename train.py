#import keras
from keras.datasets       import mnist, cifar10
from keras.models         import Sequential
from keras.layers         import Dense, Dropout, Flatten, Embedding, LSTM, GRU
from keras.utils.np_utils import to_categorical
from keras.callbacks      import EarlyStopping, Callback
from keras.layers         import Conv2D, MaxPooling2D
from keras                import backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from math import sqrt
import pandas as pd 

import logging

# Helper: Early stopping.
early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=10, verbose=0, mode='auto')
#early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=10, save_best_only = True, verbose=0, mode='auto')

def get_ctocity():
    # Set defaults.
    batch_size  = 10
    epochs      = 60
    # Get the data.
    data = pd.read_csv("/home/matheus_araujo/NetLogo 6.1.1/app/datasetevoDeepMAS.csv", header = None)
    data = data.stack().str.replace(',','.').unstack() 
    data = data.values

    X = data[1:, 1:14] 
    y = data[1:,14:16]

    # X.reshape(35976, 13, 13)
    # y.reshape(35976, 13, 2)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

    X_train = pad_sequences(X_train, maxlen = 13, dtype='float32')
    X_test = pad_sequences(X_test, maxlen = 13, dtype='float32')

    return (batch_size, X_train, X_test, y_train, y_test, epochs)

def compile_model_rnn(genome):

    # Node	
    learningrate  = genome.geneparam['learningrate' ]
    recurrentdropout  = genome.geneparam['recurrentdropout' ]
    weightunitialization  = genome.geneparam['weightunitialization' ]
    usebias  = genome.geneparam['usebias']
   # Layer
    nb_layers  = genome.geneparam['nb_layers' ]
    nb_neurons = genome.nb_neurons()
    activation = genome.geneparam['activation']
    optimizer  = genome.geneparam['optimizer' ]

    layerstype  = genome.geneparam['layerstype' ]
    layersdropout  = genome.geneparam['layersdropout' ]

    logging.info("Architecture:%f,%f,%s,%s,%s,%f,%s,%s,%s,%d" % (learningrate, recurrentdropout, weightunitialization, str(usebias), layerstype, layersdropout, str(nb_neurons), activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0 and nb_layers == 1:
            model.add(Embedding(2000, 128))
            model.add(eval(layerstype)(nb_neurons[i], activation=activation, use_bias = usebias, kernel_initializer=weightunitialization, return_sequences = False, recurrent_dropout = recurrentdropout, input_shape = (28780, 2)))
        elif i == 0 and nb_layers > 1:
            model.add(Embedding(2000, 128))
            model.add(eval(layerstype)(nb_neurons[i], activation=activation, use_bias = usebias, kernel_initializer=weightunitialization, return_sequences = True, recurrent_dropout = recurrentdropout, input_shape = (28780, 2)))	
        elif i != (nb_layers - 1):
            model.add(eval(layerstype)(nb_neurons[i], activation=activation, use_bias = usebias, kernel_initializer=weightunitialization, return_sequences = True, recurrent_dropout = recurrentdropout, input_shape = (28780, 2)))
        else:
            model.add(eval(layerstype)(nb_neurons[i], activation=activation, use_bias = usebias, kernel_initializer=weightunitialization, recurrent_dropout = recurrentdropout, input_shape = (28780, 2)))

        model.add(Dropout(layersdropout))  # hard-coded dropout for each layer

    # Output layer.
    model.add(Dense(2, activation='linear'))

    opt = getattr(optimizers, optimizer)(lr=learningrate)	

    model.compile(loss='mean_squared_error',
                    optimizer=opt,
                    metrics=['mse'])
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def train_and_score(genome, dataset, numero):

    logging.info("Getting Keras datasets")

    if dataset   == 'ctocity':
        batch_size, X_train, X_test, y_train, y_test, epochs = get_ctocity()

    logging.info("Compling Keras model")

    if dataset   == 'ctocity':
        model = compile_model_rnn(genome)
   
    history = LossHistory()

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,  
              # using early stopping so no real limit - don't want to waste time on horrible architectures
              verbose=2,
              validation_data=(X_test, y_test),
              #callbacks=[history])
              callbacks=[early_stopper])
    score = 0
    model.save('modelo ' + str(numero) + '.h5')  # creates a HDF5 file 'my_model.h5'	
    #score = model.history['val_loss']

    K.clear_session()
    #we do not care about keeping any of this in memory - 
    #we just need to know the final scores and the architecture
    return score	

def trainsimulation(genome, dataset):

    logging.info("Getting Keras datasets")

    if dataset   == 'ctocity':
        batch_size, X_train, X_test, y_train, y_test, epochs = get_ctocity()

    logging.info("Compling Keras model")

    if dataset   == 'ctocity':
        model = compile_model_rnn(genome)
   
    history = LossHistory()

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,  
              # using early stopping so no real limit - don't want to waste time on horrible architectures
              verbose=2,
              validation_data=(X_test, y_test),
              #callbacks=[history])
              callbacks=[early_stopper])

    preds = model.predict(X_test)
    score = sqrt(mean_squared_error(y_test, preds))

    model.save('model.h5')  # creates a HDF5 file 'my_model.h5'

    #score = model.history['val_loss']

    K.clear_session()
    #we do not care about keeping any of this in memory - 
    #we just need to know the final scores and the architecture
    
    return model   
