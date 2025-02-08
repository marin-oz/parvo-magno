from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf

class RandomCropAndFlip(Layer):
    def __init__(self, crop_length, crop_height, crop_width, **kwargs):
        super().__init__(**kwargs)
        self.crop_length = crop_length
        self.crop_height = crop_height
        self.crop_width = crop_width

    def call(self, inputs, training=None):
        batch, time, height, width, channels = inputs.shape

        if training:
            # Create random cropping coordinates for training
            start = tf.random.uniform(shape=[], minval=0, maxval=time-self.crop_length+1, dtype=tf.int32)
            top = tf.random.uniform(shape=[], minval=0, maxval=height-self.crop_height+1, dtype=tf.int32)
            left = tf.random.uniform(shape=[], minval=0, maxval=width-self.crop_width+1, dtype=tf.int32)

            # Randomly flip horizontally
            if tf.random.uniform([]) < 0.5:
                inputs = tf.reverse(inputs, axis=[3])
        else:
            # Create center cropping coordinates for inference
            start = (time - self.crop_length) // 2
            top = (height - self.crop_height) // 2
            left = (width - self.crop_width) // 2

        # Crop and return the tensor
        return inputs[:, start:start+self.crop_length, top:top+self.crop_height, left:left+self.crop_width, :]


class ModelConstructor:
    def __init__(self, model_name, input_shape, num_classes):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Create input layer
        inputs = Input(shape=self.input_shape)

        # Add normalization and preprocessing layers
        x = preprocessing.Rescaling(scale=1./127.5, offset=-1)(inputs)
        x = RandomCropAndFlip(32, 112, 112)(x)

        # Add specific model architecture based on model_name
        if self.model_name == "alexnet_3d":
            outputs = self.__alexnet_3d(x)
        else:
            exit('model '+self.model_name+' was not found.')

        # Create the final model
        self.model = Model(inputs, outputs)
    
    def getModel(self):
        return self.model

    # Print model summary
    def printSummary(self):
        print(self.model.summary())
    
    # Define alexnet22_48 based 3D architecture
    def __alexnet_3d(self, x):
        x = Conv3D(48, kernel_size=(5,22,22), strides=(1,4,4),
                   padding='valid', activation='relu',
                   input_shape=(32, 112, 112, 3))(x)
        x = BatchNormalization()(x)
        x = MaxPool3D(pool_size=(3,3,3), strides=(2,2,2))(x)

        x = Conv3D(256, kernel_size=(5,5,5), strides=1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool3D(pool_size=(3,3,3), strides=2)(x)
        
        x = Conv3D(384, kernel_size=(3,3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv3D(384, kernel_size=(3,3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv3D(256, kernel_size=(3,3,3), strides= 1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool3D(pool_size=(3,3,3), strides=2)(x)
        
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
       
        return x
