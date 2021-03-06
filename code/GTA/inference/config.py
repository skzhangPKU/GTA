TRAINING_STEPS = 100000
NUM_EPOCH = 2
LAYERS_NUM = 1
REGULARIZATION_RATE = 0.5
KEEP_PROB = 1
BATCH_SIZE = 128
HIDDEN_SIZE = 100
INPUT_SIZE = 249
TIME_SERIES_STEP = 6
LEARNING_RATE = 0.001
MAX_GRAD_NORM = 5
TIME_REQUIRE = 3
TRAIN_RECORD_FILE = '../input/train.tfrecords'
VAL_RECORD_FILE = '../input/validation.tfrecords'
TEST_RECORD_FILE = '../input/test.tfrecords'
CSV_PATH = '../input/adj_mx_' + str(INPUT_SIZE) + '.csv'
TEST_SAMPLE_NUMS_FIFTEEN = 2877
TRAIN_SAMPLE_NUMS_FIFTEEN = 23447
VAL_SAMPLE_NUMS_FIFTEEN = 6018
FLOW_SCALER_PATH = '../input/flow_scaler.pkl'
DATE_SCALER_PATH = '../input/date_scaler.pkl'
MODEL_NAME = 'model.ckpt'
MODEL_SAVE_PATH = '../model'
RT_MODEL_PATH = '../rt_model'
SENSOR_MX_PATH = '../input/sensor_matrix.pkl'
EMBEDDED_MX_PATH = '../input/line_128.pkl'

