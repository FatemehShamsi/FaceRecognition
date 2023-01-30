import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer


class SiameseModel:

    def make_model(self):
        inp = Input(shape=(100, 100, 3), name='Input_image')

        conv1 = Conv2D(filters=64, kernel_size=(10, 10), activation='relu')(inp)
        max1 = MaxPooling2D(64, (2, 2), padding='same')(conv1)

        conv2 = Conv2D(128, (7, 7), activation='relu')(max1)
        max2 = MaxPooling2D(64, (2, 2), padding='same')(conv2)

        conv3 = Conv2D(128, (4, 4), activation='relu')(max2)
        max3 = MaxPooling2D(64, (2, 2), padding='same')(conv3)

        conv4 = Conv2D(256, (4, 4), activation='relu')(max3)
        flat = Flatten()(conv4)
        output = Dense(4096, activation='sigmoid')(flat)

        return Model(inputs=[inp], outputs=[output], name='base_model')

    def make_siames_model(self):
        input_image = Input(name='input_image', shape=(100, 100, 3))
        validation_image = Input(name='validation_image', shape=(100, 100, 3))

        model = self.make_model()
        siames_layer = DisLayer()
        siames_layer._name = 'distance'

        distance = siames_layer(model(input_image), model(validation_image))

        classifier = Dense(1, activation='sigmoid')(distance)

        return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


class DisLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, input_em, validation):
        return tf.math.abs(input_em - validation)



if __name__ == "__main__":
    siames_net = SiameseModel().make_model()
    siames_net.summary()

