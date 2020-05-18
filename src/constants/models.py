from datetime import datetime

START_TRAIN_DATE = datetime(2017, 1, 1)
END_TRAIN_DATE = datetime(2019, 1, 1)

FREQ = "H"
NB_HOURS_PRED = 14*24

# Possibility to set learning rate and its decay factor
# For now we set them to default values from https://gluon-ts.mxnet.io/api/gluonts/gluonts.model.deepar.html
LR_DECAY_FACTOR = 0.5
LEARNING_RATE = 0.001
# For the nb of epochs, default is 100 but we set it to less to reduce training time.
# We will train with 100 epochs in the stability study, see variable DEEPAR_MAX_EPOCH_LIST_STR below.
MAX_EPOCHS = 30

IDF = "Ile-de-France"

DEEPAR_MAX_EPOCH_LIST_STR = "[20, 60, 100]"
MAX_NB_TRAININGS = "10"
NB_PRED_NUM_EVAL_SAMPLES_STUDY = "10"
