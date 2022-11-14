import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.python.keras import layers




def get_prepared_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    num_classes = 10

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def get_layers_info(model_layers):
    output = "n\ttype\tfilters\tkernel_size\tactivation\tpool_size\tstrides\n"
    for i, l in enumerate(model_layers[1:-2]):
        if type(l) == layers.MaxPooling2D or type(l) == layers.AveragePooling2D:
            output += "".join([str(i), "\t", l.__class__.__name__, "\t\t\t\t", str(l.pool_size), "\t", str(l.strides)])
        if type(l) == layers.Conv2D:
            output += "".join([str(i), "\t", l.__class__.__name__, "\t", str(l.filters), "\t", str(l.kernel_size), "\t", l.activation.__name__, "\t\t"])
        output += "\n"
    output = output[:-1]

    df = pd.DataFrame([x.split('\t') for x in output.split('\n')])
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row
    df.columns = new_header  # set the header row as the df header

    return df


colors = 'bgrcmyk'
from matplotlib.ticker import MaxNLocator


def set_plt(all_history, all_titles, all_title):
    plt.style.use('ggplot')
    fig, axis = plt.subplots(1, 2)
    fig.suptitle(all_title)
    fig.set_size_inches(13, 5)

    plt.sca(axis[0])
    for his_inx in range(len(all_history)):
        # plt.plot(all_history[his_inx].history['accuracy'], c=colors[his_inx], label="train")
        plt.plot(all_history[his_inx].history['val_accuracy'], c=colors[his_inx], label=str(all_titles[his_inx]))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    axis[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.sca(axis[1])
    for his_inx in range(len(all_history)):
        # plt.plot(all_history[his_inx].history['loss'], c=colors[his_inx], label="train")
        plt.plot(all_history[his_inx].history['val_loss'], c=colors[his_inx], label=str(all_titles[his_inx]))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    axis[0].xaxis.set_major_locator(MaxNLocator(integer=True))
