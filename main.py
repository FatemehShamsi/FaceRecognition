import tensorflow as tf
from prepare_data import *
from model import *
from tensorflow.python.keras.metrics import Recall, Precision

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001
siames_net = SiameseModel().make_model()
train_data, test_data = data_loader.train_test_loader()


@tf.function
def train_step(batch):
    with tf.gradients as tape:
        X = batch[:2]
        y = batch[2]

        y_hat = siames_net(X)
        loss = binary_cross_loss(y, y_hat)
        print(loss)

    grad = tape.gradient(loss, siames_net.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siames_net.trainable_variables))


def train(data, Epochs):
    for epoch in range(1, Epochs + 1):
        print('\n Epoch {}/{}'.format(epoch, Epochs))

        r = Recall()
        p = Precision()

        for idx, batch in enumerate(data):
            loss = train_step(batch)
            y_hat = siames_net.predict(batch[:2])
            r.update_state(batch[2], y_hat)
            p.update_state(batch[2], y_hat)

        print(loss.numpy(), r.result().numpy(), p.result().numpy())


Epochs = 50
train(train_data, Epochs)
siames_net.save('siamesemodelv2.h5')
