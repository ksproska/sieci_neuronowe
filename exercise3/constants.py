import keras.metrics.metrics

OPT_ADAM = "adam"
OPT_ADAGRAD = "adagrad"
OPT_ADADELTA = "adadelta"
OPT_RMS_PROP = "rmsprop"
OPT_SGD = "sgd"

ACT_RELU = "relu"
ACT_TANH = "tanh"
ACT_SIGMOID = "sigmoid"

TP = keras.metrics.metrics.TruePositives()
TN = keras.metrics.metrics.TrueNegatives()
FP = keras.metrics.metrics.FalsePositives()
FN = keras.metrics.metrics.FalseNegatives()
AUC = keras.metrics.metrics.AUC()
