import argparse
import sys
import os
import csv

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    inputVector = np.load("../SVM/trainVector.npy")
    embeddingVector = tf.Variable(inputVector, trainable=False, name="embedding")
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(FLAGS.log_dir + "/model.ckpt"))
    metadata_file = FLAGS.log_dir + "/metadata.tsv"
    labelFile = "../SVM/trainLabel.csv"
    with open(metadata_file, "w") as metadata:
        with open(labelFile, "r") as lFile:
            reader = csv.reader(lFile)
            for row in reader:
                metadata.write('{}\n'.format(row))
    writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    config = projector.ProjectorConfig
    embeddings = config.embeddings.add()
    embeddings.tensor_name = 'embedding:0'
    embeddings.metadata_path = os.path.join(FLAGS.log_dir + '/metadata.tsv')
    projector.visualize_embeddings(writer, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default="../VISUALIZATION/tensorboard")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
