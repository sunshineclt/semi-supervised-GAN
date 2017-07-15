import keras.backend as K
from keras import regularizers
from keras.layers import Input, BatchNormalization, Dense, GaussianNoise
from keras.models import Model


class Generator:
    def __init__(self):
        # generator
        self.input = Input([100])
        layer = Dense(500, activation=K.softplus)(self.input)
        layer = BatchNormalization()(layer)
        layer = Dense(500, activation=K.softplus)(layer)
        layer = BatchNormalization()(layer)
        self.output = Dense(28 ** 2, activation=K.sigmoid, kernel_regularizer=regularizers.l2())(layer)
        self.model = Model(inputs=self.input, outputs=self.output)


class Discriminator:
    def __init__(self):
        # discriminator
        self.input = Input([28 ** 2])
        layer = GaussianNoise(stddev=0.3)(self.input)
        layer = Dense(1000)(layer)
        layer = GaussianNoise(stddev=0.5)(layer)
        layer = Dense(500)(layer)
        layer = GaussianNoise(stddev=0.5)(layer)
        layer = Dense(250)(layer)
        layer = GaussianNoise(stddev=0.5)(layer)
        layer = Dense(250)(layer)
        layer = GaussianNoise(stddev=0.5)(layer)
        layer = Dense(250)(layer)
        self.feature = Model(inputs=self.input, outputs=layer)
        layer = GaussianNoise(stddev=0.5)(layer)
        self.output = Dense(10)(layer)
        self.model = Model(inputs=self.input, outputs=self.output)
