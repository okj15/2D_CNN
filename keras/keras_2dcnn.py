#  -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import tensorflow as tf
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras import backend as K


class CNN_Model():

    def __init__(self, X_train, y_train, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_history(self, history, outdir):

        plt.figure()
        plt.plot(history.history['acc'], marker='.')
        plt.plot(history.history['val_acc'], marker='.')
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.grid()
        plt.legend(['train', 'test'], loc='best')
        plt.savefig('{0}acc.png'.format(outdir))
        plt.clf()

        plt.figure()
        plt.plot(history.history['loss'], marker='.')
        plt.plot(history.history['val_loss'], marker='.')
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(['train', 'test'], loc='best')
        plt.savefig('{0}loss.png'.format(outdir))
        plt.clf()

    def cnn_model(self, inputs, fil=2, st=2, pool=2):

        x = Conv2D(32, kernel_size=(fil, fil), padding='valid', input_shape=(self.X_train.shape[1], self.X_train.shape[2], 1), strides=(st, st), activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(pool, pool), padding='valid')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        return x

    def training(self):

        Inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2], 1))
        x = self.cnn_model(Inputs)

        feature = Dense(256, activation='relu')(x)
        Outputs = Dense(10, activation='softmax')(feature)

        model = Model(inputs=Inputs, outputs=Outputs)
        plot_model(model, show_shapes=True, to_file='model.png')

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        config = tf.ConfigProto(
            inter_op_parallelism_threads=6,
            intra_op_parallelism_threads=6)
        session = tf.Session(config=config)
        K.set_session(session)

        history = model.fit(self.X_train, self.y_train,
                            batch_size=100,
                            epochs=20,
                            verbose=1,
                            validation_split=0.1,
                            shuffle=True)

        self.plot_history(history, './')

        loss, acc = model.evaluate(self.X_test, self.y_test, verbose=0)

        print('Test loss:', loss)
        print('Test acc:', acc)

        K.clear_session()


def main():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], 1)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    CM = CNN_Model(X_train, y_train, X_test, y_test)
    CM.training()


if __name__ == '__main__':
    main()
