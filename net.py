from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import MaxPooling2D, Activation, Dropout, Flatten, Dense, SeparableConv2D,Conv2D ,initializers
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from utility import *

def train_model(train_generator, validation_generator, learning_rate, decay, n_epochs, batch, optimization_type,folder_weights):

    print('[INFO] training model...')
    model=Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(48, 48, 3),kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(64,kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1, kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Activation("sigmoid"))
    #print(model.summary())

    opt = Adam(lr=learning_rate, decay=decay)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    checkpointer = ModelCheckpoint(
        folder_weights + '/{}_weights_{}_{}.hdf5'.format(optimization_type, learning_rate, decay), verbose=1, mode=min,
        save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logdir/' + optimization_type, batch_size=batch, update_freq='epoch')
    callbacks = [checkpointer, tensorboard]
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch,
        verbose=2,
        callbacks=callbacks)

    valLosses = history.history['val_loss']
    valAccuracys = history.history['val_acc']
    trainLosses = history.history['loss']
    trainAccuracys = history.history['acc']
    val_loss = min(valLosses)

    best_index = valLosses.index(val_loss)
    val_acc = valAccuracys[best_index]
    train_loss = trainLosses[best_index]
    train_acc = trainAccuracys[best_index]
    best_epoch = best_index + 1

    print('min val_loss: {} , epoch: {} , accuracy: {} '.format(val_loss,best_epoch,val_acc))
    #plot_loss_trainVal(history)
    #plot_accuracy_trainVal(history)

    return train_acc, train_loss, val_acc, val_loss, valLosses, best_epoch


def test_model(model, test_generator):
    print("[INFO] Test...")
    test_generator.reset()
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples)
    print('test_acc:{}, test_loss:{} '.format(test_acc, test_loss))
    return test_acc, test_loss
