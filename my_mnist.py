import mnist
import argparse

import tensorflow as tf
import sys
from mnist import dataset
import os

LEARNING_RATE = 1e-4

def create_model(data_format):
    """Model to recognize digits in the MNIST dataset."""
    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
        assert data_format == 'channels_last'
        input_shape = [28, 28, 1]

    l = tf.keras.layers
    max_pool = l.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)
    # The model consists of a sequential chain of layers, so tf.keras.Sequential
    # (a subclass of tf.keras.Model) makes for a compact description.
    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(28 * 28,)),
            l.Conv2D(
                32,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Conv2D(
                64,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.Dropout(0.4),
            l.Dense(2)
        ])



def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = create_model(params['data_format'])
    image = features
    if isinstance(image, dict):
        image = features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params.get('multi_gpu'):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))

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
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy':
                    tf.metrics.accuracy(
                        labels=labels, predictions=tf.argmax(logits, axis=1)),
            })


def run_mnist():
    model_function = model_fn

    num_gpus = FLAGS.num_gpus
    multi_gpu = num_gpus > 1

    if multi_gpu:
        #TODO Validate that the batch size can be split into devices.
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_fn, loss_reduction=tf.losses.Reduction.MEAN,
            devices=["/device:GPU:%d" % d for d in range(num_gpus)])

    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=FLAGS.model_dir,
        params={
            'data_format': data_format,
            'multi_gpu': multi_gpu
        })

    # Set up training and evaluation input functions.
    def train_input_fn():
        """Prepare data for training."""

        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes use less memory. MNIST is a small
        # enough dataset that we can easily shuffle the full epoch.
        directory  = os.path.join(FLAGS.data_dir, 'train')
        ds = dataset.image_train(directory)
        ds = ds.cache().shuffle(buffer_size=8800).batch(FLAGS.batch_size)

        # Iterate through the dataset a set number (`epochs_between_evals`) of times
        # during each training session.
        ds = ds.repeat(FLAGS.epochs_between_evals)
        return ds

    def eval_input_fn():
        directory = os.path.join(FLAGS.data_dir, 'eval')
        return dataset.image_eval(directory).batch(
            FLAGS.batch_size).make_one_shot_iterator().get_next()

    # Train and evaluate model.
    for _ in range(FLAGS.train_epochs // FLAGS.epochs_between_evals):
        mnist_classifier.train(input_fn=train_input_fn)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print('\nEvaluation results:\n\t%s\n' % eval_results)

    # Export the model
    if FLAGS.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
        mnist_classifier.export_savedmodel(FLAGS.export_dir, input_fn)



def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_mnist()

    # ds = dataset.train('E://tmp//mnist_data')
    # ds = ds.take(10)
    # iterator = ds.make_one_shot_iterator()
    # one_element = iterator.get_next()
    # with tf.Session() as sess:
    #     try:
    #         while True:
    #             print(sess.run(one_element))
    #     except tf.errors.OutOfRangeError:
    #         print('end!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
        '--data_dir', type=str, default="E://maotaiSet//trainval-5",
        help='path to model weight file'
    )

    parser.add_argument(
        '--model_dir', type=str, default="/tmp/mnist_model",
        help='The location of the model checkpoint files'
    )

    parser.add_argument(
        '--log_dir', type=str, default="/tmp/log_dir",
        help='Directory to put the log data'
    )


    parser.add_argument(
        '--clean', type=bool, default=False,
        help='If set, model_dir will be removed if it exists.'
    )

    parser.add_argument(
        '--train_epochs', type=int, default=40,
        help='The number of epochs used to train.'
    )

    parser.add_argument(
        '--epochs_between_evals', type=int, default=1,
        help='The number of training epochs to run between evaluations.'
    )

    parser.add_argument(
        '--stop_threshold', type=float, default=None,
        help='If passed, training will stop at the earlier of train_epochs and '
             'when the evaluation metric is greater than or equal to stop_threshold.'
    )

    parser.add_argument(
        '--batch_size', type=int, default=100,
        help='Batch size for training and evaluation.'
    )

    parser.add_argument(
        '--num_gpus', type=int, default=1 if tf.test.is_gpu_available() else 0,
        help='How many GPUs to use with the DistributionStrategies API. '
    )

    parser.add_argument(
        '--export_dir', type=str, default=None,
        help='If set, a SavedModel serialization of the model will '
             'be exported to this directory at the end of training. '
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)








