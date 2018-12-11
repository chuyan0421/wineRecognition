import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from os.path import isfile
from keras.models import load_model
import argparse
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from keras.layers import Input
import os
import numpy as np


def get_class():
    classes_path = os.path.expanduser(FLAGS.classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors():
    anchors_path = os.path.expanduser(FLAGS.anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def build_model():

    global yolo_score
    yolo_score = 0.3

    global yolo_iou
    yolo_iou = 0.45

    global graph
    graph = tf.get_default_graph()

    yolo_anchors = get_anchors()
    class_names = get_class()

    num_anchors = len(yolo_anchors)
    num_classes = len(class_names)

    global model
    global input_image_shape
    global boxes
    global scores
    global classes

    model = yolo_body(Input(shape=(416, 416, 3)), num_anchors // 3, num_classes)
    model.load_weights(FLAGS.model_path)
    print('{} model, anchors, and classes loaded.'.format(FLAGS.model_path))

    input_image_shape = K.placeholder(shape=(None, 2))

    boxes, scores, classes = yolo_eval(model.output, yolo_anchors,
                                       num_classes, input_image_shape,
                                       score_threshold=yolo_score, iou_threshold=yolo_iou)


def save_model_to_serving(export_version, export_path):

    with graph.as_default():
        # print(model.input, model.output)
        print(model.input.shape)
        print(input_image_shape.shape)
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={
                'image': model.input,
                'size': input_image_shape
            },
            outputs={
                'boxes': boxes,
                'scores': scores,
                'classes': classes
            }
        )
        exported_path = os.path.join(
            tf.compat.as_bytes(export_path),
            tf.compat.as_bytes(str(export_version))
        )
        builder = tf.saved_model.builder.SavedModelBuilder(exported_path)
        # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'classify': signature,
            },
            main_op=tf.tables_initializer(),
            strip_default_attrs=True
            # legacy_init_op=legacy_init_op
        )
        builder.save()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path', type=str, default='model_data/yolo_weights.h5',
        help='path to model weight file'
    )

    parser.add_argument(
        '--anchors_path', type=str, default='model_data/yolo_anchors.txt',
        help='path to anchors file'
    )

    parser.add_argument(
        '--classes_path', type=str, default='model_data/coco_classes.txt',
        help='path to coco classed file'
    )

    parser.add_argument(
        '--yolo_saved_model', type=str, default='/tmp/yolo_saved_model',
        help='path to saved model'
    )


    FLAGS, unparsed = parser.parse_known_args()
    export_path = FLAGS.yolo_saved_model
    if not os.path.exists(export_path):
        os.mkdir(export_path)

    build_model()
    save_model_to_serving('1', export_path)



