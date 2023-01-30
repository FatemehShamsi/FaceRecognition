import os
import shutil
import urllib.request
import cv2
import uuid
import tensorflow as tf


def preprocess(path_file):
    byte_image = tf.io.read_file(path_file)
    img = tf.io.decode_jpeg(byte_image)

    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


class DataCreate:
    def __init__(self, neg_path, pos_path, anc_path):
        self.neg_path = neg_path
        self.pos_path = pos_path
        self.anc_path = anc_path

        if not os.path.exists('data'):
            self.create_dir()
        if not os.path.exists('lfw'):
            self.get_facial_image()
        self.collect_pos_anc()

    def create_dir(self):
        os.makedirs(self.pos_path)
        os.makedirs(self.neg_path)
        os.makedirs(self.anc_path)

    def collect_neg_image(self):
        for directory in os.listdir('lfw'):
            for file in os.listdir(os.path.join('lfw', directory)):
                ex_path = os.path.join('lfw', directory, file)
                new_path = os.path.join(self.neg_path, file)
                os.replace(ex_path, new_path)

    def get_facial_image(self):
        url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
        urllib.request.urlretrieve(url, 'lfw.tgz')
        shutil.unpack_archive('lfw.tgz')
        self.collect_neg_image()

    def collect_pos_anc(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = frame[120:120 + 250, 200:200 + 250, :]
            cv2.imshow('Image Collection', frame)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                img_name = os.path.join(self.anc_path, '{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(img_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('p'):
                img_name = os.path.join(self.pos_path, '{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(img_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


class DataLoader:
    def __init__(self, ):
        self.neg_path = os.path.join('data', 'negative')
        self.pos_path = os.path.join('data', 'positive')
        self.anc_path = os.path.join('data', 'anc')

        data_create = DataCreate(self.neg_path, self.pos_path, self.anc_path)

        self.positive = tf.data.Dataset.list_files(self.pos_path + '\*.jpg').take(30)
        self.negative = tf.data.Dataset.list_files(self.neg_path + '\*.jpg').take(30)
        self.anchor = tf.data.Dataset.list_files(self.anc_path + '\*.jpg').take(30)

    def zip_pos_neg_data(self):
        # create labelled dataset
        positive_data = tf.data.Dataset.zip(
            (self.anchor, self.positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(self.anchor)))))
        negative_data = tf.data.Dataset.zip(
            (self.anchor, self.negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(self.anchor)))))
        data = positive_data.concatenate(negative_data)
        return data

    def twin_preprocess(self, input_image, validation_image, label):
        pre_data = (preprocess(input_image), preprocess(validation_image), label)
        return pre_data

    def train_test_loader(self):
        data = self.zip_pos_neg_data()
        data = data.map(self.twin_preprocess)
        data = data.cache()
        data = data.shuffle(buffer_size=1024)

        train_data = data.take(round((len(data) * 0.7)))
        train_data = train_data.batch(16)
        train_data = train_data.prefetch(8)

        test_data = data.skip(round((len(data) * 0.7)))
        test_data = test_data.batch(16)
        test_data = test_data.prefetch(8)

        return train_data, test_data


if __name__ == "__main__":
    data_loader = DataLoader()
    train, test = data_loader.train_test_loader()
