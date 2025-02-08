from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomCrop
import tensorflow as tf

class HalfConv2D(Layer):
    def __init__(self, filters, kernel_size, trainable=[True, True], **kwargs):
        super(HalfConv2D, self).__init__()
        self.filters = filters // 2
        self.kernel_size = kernel_size
        self.kwargs = kwargs
        self.conv_0 = Conv2D(self.filters, self.kernel_size, trainable=trainable[0], **kwargs)
        self.conv_1 = Conv2D(self.filters, self.kernel_size, trainable=trainable[1], **kwargs)

    def build(self, input_shape):
        self.conv_0.build(input_shape)
        self.conv_1.build(input_shape)

    def call(self, inputs):
        conv_0_out = self.conv_0(inputs)
        conv_1_out = self.conv_1(inputs)
        conv_out = Concatenate()([conv_0_out, conv_1_out])
        return conv_out
    
    def get_weights(self):
        conv_0_weights = self.conv_0.get_weights()
        conv_1_weights = self.conv_1.get_weights()
        concatenated_kernel = tf.concat([conv_0_weights[0], conv_1_weights[0]], axis=-1)
        concatenated_bias = tf.concat([conv_0_weights[1], conv_1_weights[1]], axis=0)
        return [concatenated_kernel, concatenated_bias]
    
    def set_weights(self, weights):
        conv_0_kernel = weights[0][:, :, :, :self.filters]
        conv_1_kernel = weights[0][:, :, :, self.filters:]
        conv_0_bias = weights[1][:self.filters]
        conv_1_bias = weights[1][self.filters:]
        self.conv_0.set_weights([conv_0_kernel, conv_0_bias])
        self.conv_1.set_weights([conv_1_kernel, conv_1_bias])

class ModelConstructor:
    def __init__(self, model_name, input_shape, num_classes, half_conv_train=[True, True]):
        """
        Constructor for the ModelConstructor class.
        Args:
        model_name (str): name of the model to be used
        input_shape (tuple): shape of the input data
        num_classes (int): number of classes in the dataset
        half_conv_train (list): list of two boolean values indicating whether the first and 
                                second half of the first convolutional layer are trainable
                                only used for alexnet22_48_half (biomimetic v4)
        """
        self.model_name = model_name

        # Check if model_name is valid
        if self.model_name not in ["alexnet", "alexnet22", "alexnet22_48", "alexnet22_48_half"]:
            exit('model '+self.model_name+' was not found')
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Create input layer
        inputs = Input(shape=self.input_shape)

        # Add normalization and preprocessing layers
        x = Rescaling(scale=1./127.5, offset=-1)(inputs)
        x = RandomFlip("horizontal")(x)

        # Add specific model architecture based on model_name
        if self.model_name == "alexnet":
            x = RandomCrop(227, 227)(x)
            outputs = self.__alexnet(x)
        
        elif self.model_name == "alexnet22":
            x = RandomCrop(227, 227)(x)
            outputs = self.__alexnet22(x)

        elif self.model_name == "alexnet22_48":
            x = RandomCrop(227, 227)(x)
            outputs = self.__alexnet22_48(x)

        elif self.model_name == "alexnet22_48_half":
            x = RandomCrop(227, 227)(x)
            outputs = self.__alexnet22_48_half(x, half_conv_train)
        
        # Create the final model
        self.model = Model(inputs, outputs)

    def getModel(self):
        return self.model

    # Print model summary
    def printSummary(self):
        print(self.model.summary())

    # Define AlexNet model architecture
    def __alexnet(self, x):
        x = Conv2D(96, kernel_size=(11,11), strides=4,
                   padding='valid', activation='relu',
                   input_shape=(227, 227, 3))(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
        x = Conv2D(256, kernel_size=(5,5), strides=1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
        x = Conv2D(384, kernel_size=(3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv2D(384, kernel_size=(3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv2D(256, kernel_size=(3,3), strides= 1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides= (2,2))(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
       
        return x
    
    # Define AlexNet architecture with filter size 22 x 22
    def __alexnet22(self, x):
        x = Conv2D(96, kernel_size=(22,22), strides=4,
                   padding='valid', activation='relu',
                   input_shape=(227, 227, 3))(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
        x = Conv2D(256, kernel_size=(5,5), strides=1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
        x = Conv2D(384, kernel_size=(3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv2D(384, kernel_size=(3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv2D(256, kernel_size=(3,3), strides= 1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides= (2,2))(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
       
        return x
    
    # Define AlexNet architecture with filter size 22 x 22 and 48 filters
    def __alexnet22_48(self, x):
        x = Conv2D(48, kernel_size=(22,22), strides=4,
                   padding='valid', activation='relu',
                   input_shape=(227, 227, 3))(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
        x = Conv2D(256, kernel_size=(5,5), strides=1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
        x = Conv2D(384, kernel_size=(3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv2D(384, kernel_size=(3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv2D(256, kernel_size=(3,3), strides= 1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides= (2,2))(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
    
        return x
    
    # Define AlexNet architecture with filter size 22 x 22 and 48 filters HalfConv2D as the first layer
    def __alexnet22_48_half(self, x, trainable=[True, True]):
        x = HalfConv2D(48, kernel_size=(22,22), trainable=trainable, strides=4,
                       padding='valid', activation='relu',
                       input_shape=(227, 227, 3))(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
        x = Conv2D(256, kernel_size=(5,5), strides=1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
        x = Conv2D(384, kernel_size=(3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv2D(384, kernel_size=(3,3), strides=1,
                   padding='same', activation='relu')(x)
        x = Conv2D(256, kernel_size=(3,3), strides= 1,
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3,3), strides= (2,2))(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
    
        return x

    