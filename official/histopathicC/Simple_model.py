"""Convolutional Neural Network Estimator for Simple_model, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
import random
import pickle
import modelarchs

# read mean value
readpath = 'dataset/mean_v'
with open(readpath, 'rb') as fp:
    mean_value = pickle.load(fp)

LEARNING_RATE = 1e-4

def define_flags():
    flags_core.define_base()
    flags_core.define_performance(num_parallel_calls=False)
    flags_core.define_image()
    flags.adopt_module_key_flags(flags_core)
    flags_core.set_defaults(data_dir='Simple_model_data',
                            model_dir='Simple_model_model',
                            batch_size=128,
                            train_epochs=100)


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = modelarchs.VGGlike(params['data_format'])
    model.summary()
    image = features
    if isinstance(image, dict):
        image = features['image']

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        logits = model(image, training=True)
        labels = tf.reshape(labels, (-1, 1))
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
        """
        how tf.accuracy works.
        we have a batch (rows) with labels(?, 1) and logits(?, n_of_classes).
        We call tf.argmax so we can find the index of max in each row, which corresponds
        to the label
        """
        result = tf.cast(tf.greater(logits, 0.5), tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=result)

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        labels = tf.reshape(labels, (-1, 1))
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

        result = tf.cast(tf.greater(logits, 0.5), tf.int32)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy':
                    tf.metrics.accuracy(
                        labels=labels, predictions=result),
            })


def run_Simple_model(flags_obj):
    """Run Simple_model training and eval loop.

    Args:
      flags_obj: An object containing parsed flag values.
    """
    model_helpers.apply_clean(flags_obj)
    model_function = model_fn

    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
        intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
        allow_soft_placement=True)

    distribution_strategy = distribution_utils.get_distribution_strategy(
        flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy, session_config=session_config)

    data_format = 'channels_first'

    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=flags_obj.model_dir,
        config=run_config,
        params={
            'data_format': data_format,
        })


    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser_train(record):
        keys_to_features = {
            "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.image.decode_jpeg(parsed["image_raw"])
        image = tf.reshape(image, [64, 64, 3])
        # subtract mean
        mean = tf.convert_to_tensor(mean_value, dtype=tf.uint8)
        image = image - mean

        image = tf.image.random_crop(image, [48, 48, 3])
        image = tf.image.random_flip_left_right(image)  # flip image randomly with a 50% chance
        image = tf.image.random_brightness(image, max_delta=0.3)  # random brightness
        l = [1, 2, 3]
        deg = random.choice(l)
        image = tf.image.rot90(image, k=deg)
        image = tf.transpose(image, perm=[2, 0, 1])  # channels first
        image = tf.cast(image, tf.int32)
        image = tf.truediv(image, 255)
        image = tf.cast(image, tf.float32)
        label = tf.cast(parsed["label"], tf.int32)

        return {"image": image}, label

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser_eval(record):
        keys_to_features = {
            "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.image.decode_jpeg(parsed["image_raw"])
        image = tf.reshape(image, [64, 64, 3])
        # subtract mean
        mean = tf.convert_to_tensor(mean_value, dtype=tf.uint8)
        image = image - mean

        image = tf.image.random_crop(image, [48, 48, 3])
        image = tf.transpose(image, perm=[2, 0, 1])  # channels first
        image = tf.cast(image, tf.int32)
        image = tf.truediv(image, 255)
        image = tf.cast(image, tf.float32)
        label = tf.cast(parsed["label"], tf.int32)

        return {"image": image}, label

    # Set up training and evaluation input functions.
    def train_input_fn():
        """Prepare data for training."""
        train_tfrecord = 'dataset/train_images.tfrecords'
        dataset = tf.data.TFRecordDataset(train_tfrecord)

        # Use `Dataset.map()` to build a pair of a feature dictionary and a label
        # tensor for each example.
        dataset = dataset.map(parser_train)
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.batch(flags_obj.batch_size)

        # Iterate through the dataset a set number (`epochs_between_evals`) of times
        # during each training session.
        dataset = dataset.repeat(flags_obj.epochs_between_evals)

        return dataset

    def eval_input_fn():
        filenames = ["dataset/val_images.tfrecords"]
        dataset = tf.data.TFRecordDataset(filenames)

        # Use `Dataset.map()` to build a pair of a feature dictionary and a label
        # tensor for each example.
        dataset = dataset.map(parser_eval)
        dataset = dataset.batch(flags_obj.batch_size)
        # repeat:restarts dataset when reaches the end so here is not needed
        # dataset = dataset.repeat()

        return dataset

    # Set up hook that outputs training logs every 100 steps.
    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks, model_dir=flags_obj.model_dir,
        batch_size=flags_obj.batch_size)


    # Train and evaluate model.
    for _ in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
        classifier.train(input_fn=train_input_fn, hooks=train_hooks)
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print('\nEvaluation results:\n\t%s\n' % eval_results)

        if model_helpers.past_stop_threshold(flags_obj.stop_threshold,
                                             eval_results['accuracy']):
            break

    # Export the model
    if flags_obj.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 3, 48, 48])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
        classifier.export_savedmodel(flags_obj.export_dir, input_fn,
                                           strip_default_attrs=True)


def main(_):
    run_Simple_model(flags.FLAGS)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    absl_app.run(main)
