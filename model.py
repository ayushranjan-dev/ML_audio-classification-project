import tensorflow as tf
from keras import layers, models
from config import N_MELS



import tensorflow as tf
from keras import layers, models

def build_cnn_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Keep dropout at 0.5
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Keep dropout at 0.5

    if num_classes == 2:
        output_activation = 'sigmoid'
        units = 1
    else:
        output_activation = 'softmax'
        units = num_classes

    outputs = layers.Dense(units, activation=output_activation)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="car_sound_cnn_simple")
    return model