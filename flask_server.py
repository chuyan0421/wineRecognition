from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf

import os
import argparse
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from keras.layers import Input
from yolo3.utils import letterbox_image
from keras import backend as K


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)


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

    global yolo_anchors
    yolo_anchors = get_anchors()

    global yolo_classes
    yolo_classes = get_class()

    global yolo_score
    yolo_score = 0.3

    global yolo_iou
    yolo_iou = 0.45

    num_anchors = len(yolo_anchors)
    num_classes = len(yolo_classes)

    global graph
    graph = tf.get_default_graph()

    global sess
    sess = K.get_session()

    global model

    global input_image_shape
    global boxes
    global scores
    global classes

    model = yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
    model.load_weights(FLAGS.model_path)
    print('{} model, anchors, and classes loaded.'.format(FLAGS.model_path))

    input_image_shape = K.placeholder(shape=(2,))

    boxes, scores, classes = yolo_eval(model.output, yolo_anchors,
                                       num_classes, input_image_shape,
                                       score_threshold=yolo_score, iou_threshold=yolo_iou)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):

            # read the image in PIL format
            image_raw = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image_raw))


            boxed_image = letterbox_image(image, tuple(reversed((416, 416))))

            image_data = np.array(boxed_image, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            with graph.as_default():
                out_boxes, out_scores, out_classes = sess.run(
                    [boxes, scores, classes],
                    feed_dict={
                        model.input: image_data,
                        input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0
                    })

                print('Found {} boxes for {}'.format(len(out_boxes), 'img'))


                for i, c in reversed(list(enumerate(out_classes))):
                    predicted_class = yolo_classes[c]
                    box = out_boxes[i]
                    score = out_scores[i]

                    label = '{} {:.2f}'.format(predicted_class, score)

                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                    print(label, (left, top), (right, bottom))
                    


                data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
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

    FLAGS, unparsed = parser.parse_known_args()

    build_model()
    app.run()
