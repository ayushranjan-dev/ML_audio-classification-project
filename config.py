
DATA_DIR = "data"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

SAMPLE_RATE = 22050
DURATION = 3.0            
SAMPLES = int(SAMPLE_RATE * DURATION)

N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-5
MODEL_DIR = "saved_models"
MODEL_NAME = "car_sound_cnn.h5"

SEED = 42