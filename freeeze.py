import mobilenet_v1
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
import mobilenet_v1_prune
flags = tf.app.flags

flags.DEFINE_integer('batch_size', None, 'Batch size')
flags.DEFINE_integer('num_classes', 3755, 'Number of classes to distinguish')

flags.DEFINE_integer('image_size', 64, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 0.5, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', True, 'Quantize training')
flags.DEFINE_string('fine_tune_checkpoint', '',
                    'Checkpoint from which to start finetuning.')
flags.DEFINE_string('checkpoint_dir', './tmp',
                    'Directory for writing training checkpoints and logs')

tf.app.flags.DEFINE_string('output_file', './model/tmp/mobile_net_v0.5_prune.pb',
                           'Where to save the resulting file to.')
FLAGS = flags.FLAGS


def freeze():
    with tf.Graph().as_default() as graph:
        placeholder = tf.placeholder(name='input',
                                     dtype=tf.float32,
                                     shape=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1])
        scope = mobilenet_v1.mobilenet_v1_arg_scope(
            is_training=False, weight_decay=0.0)
        with slim.arg_scope(scope):
            mobilenet_v1.mobilenet_v1(placeholder,is_training=False,
                                        depth_multiplier=FLAGS.depth_multiplier,
                                        num_classes=FLAGS.num_classes)

        if FLAGS.quantize:
            tf.contrib.quantize.create_eval_graph()
        graph_def = graph.as_graph_def()
        with tf.gfile.GFile(FLAGS.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())

def freeze_with_prune():
    with tf.Graph().as_default() as graph:
        placeholder = tf.placeholder(name='input',
                                     dtype=tf.float32,
                                     shape=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1])
        scope = mobilenet_v1.mobilenet_v1_arg_scope(
            is_training=False, weight_decay=0.0)
        with slim.arg_scope(scope):
            mobilenet_v1_prune.mobilenet_v1(placeholder,is_training=False,
                                        depth_multiplier=FLAGS.depth_multiplier,
                                        num_classes=FLAGS.num_classes)

        if FLAGS.quantize:
            tf.contrib.quantize.create_eval_graph()
        graph_def = graph.as_graph_def()
        with tf.gfile.GFile(FLAGS.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())
def main(_):
    #freeze()
    freeze_with_prune()
if __name__ == '__main__':
  tf.app.run()