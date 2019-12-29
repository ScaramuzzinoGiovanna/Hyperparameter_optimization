import matplotlib.pyplot as plt
import os
import numpy as np
import csv
from keras.preprocessing.image import ImageDataGenerator

header_hyp = ['Learning Rate', 'Decay', 'Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss', 'Epoch']
header_loss = ['Learning Rate', 'Decay', 'Val Loss', 'Losses']


def create_folder(name, path=None):
    if path == None:
        path_folder = name
    else:
        path_folder = os.path.join(path, name)
    try:
        os.makedirs(path_folder)
    except FileExistsError:
        print(' directory {} already exist'.format(path_folder))
        pass
    except OSError:
        print('creation of the directory {} failed'.format(path_folder))
        pass
    else:
        print("Succesfully created the directory {} ".format(path_folder))

    return path_folder


def create_three_folder(path, name1, name2, name3):
    path_folder1 = create_folder(name1, path)
    path_folder2 = create_folder(name2, path)
    path_folder3 = create_folder(name3, path)

    return path_folder1, path_folder2, path_folder3


def create_csv(file_csv, header):
    with open(file_csv, mode='a') as file:
        file_csv = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(header)


def saveHyp(file_csv, learning_rate, decay, val_acc, val_loss, epoch, train_acc, train_loss):
    with open(file_csv, mode='a') as file:
        file_csv = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(
            [str(round(learning_rate, 6)), str(round(decay, 6)), str(round(train_acc, 4)), str(round(train_loss, 4)),
             str(round(val_acc, 4)), str(round(val_loss, 4)), str(epoch)])


def saveLoss(file_csv, learning_rate, decay, val_loss, losses):
    with open(file_csv, mode='a') as file:
        file_csv = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow([str(round(learning_rate, 6)), str(round(decay, 6)), str(round(val_loss, 4)), str(losses)])


def create_folder_and_csv(folder_out, optimization_type):
    folder_opt = create_folder( optimization_type,folder_out)
    folder_csv, folder_plots, folder_weights = create_three_folder(folder_opt, 'csv', 'plots', 'weights')

    path_loss = os.path.join(folder_csv, optimization_type + '_losses.csv')
    path_hyp = os.path.join(folder_csv, optimization_type + '_hyperparameters.csv')
    create_csv(path_hyp, header_hyp)
    create_csv(path_loss, header_loss)

    return path_loss, path_hyp, folder_csv, folder_plots, folder_weights


def sortLoss(val):
    return val[2]


def plot_loss(vect, type, n_epochs, folder_plot):
    vect.sort(key=sortLoss)
    plt.style.use("ggplot")
    plt.figure()
    for i in range(3):  # best 3 loss
        lr, wd, val_loss, losses = vect[i]
        plt.plot(range(0, n_epochs), losses, label='lr:' + str(round(lr, 6)) + ' wd:' + str(round(wd, 6)))
    plt.title("Validation Loss ")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    axes = plt.gca()
    axes.set_xlim([0, n_epochs - 1])
    plt.xticks(np.arange(0, n_epochs, 1))
    plt.savefig(folder_plot + '/plot_' + type + '.png')


def extractor_vect(vect, index, n_eval):
    v = []
    for elem in vect:
        v.append(elem[index])
    return v


def plot_loss_trainVal(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_accuracy_trainVal(history):
    acc_values = history.history['acc']
    val_acc_values = history.history['val_acc']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'r', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def preprocessing(train_dir, test_dir, val_split=0.2, batch=20):
    IMAGE_WIDTH = IMAGE_HEIGHT = 48

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=val_split)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=batch,
        subset='training',
        shuffle=True,
        color_mode="rgb",
        class_mode='binary', seed=13)

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=batch,
        subset='validation',
        color_mode="rgb",
        class_mode='binary', seed=13)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=1,
        color_mode="rgb",
        class_mode='binary', shuffle=False)
    return train_generator, validation_generator, test_generator


def plot_loss_trainVal(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_accuracy_trainVal(history):
    acc_values = history.history['acc']
    val_acc_values = history.history['val_acc']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
