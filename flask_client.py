# import the necessary packages
import requests
import argparse
import os
from PIL import Image

class YOLO_CLIENT(object):
    _defaults = {
        "address_port": 'localhost:5000',
        "signature": 'predict',
        "prediction_path": './prediction',
        "image_path": '5.jpg'
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

def yolo_request(image_path=YOLO_CLIENT.get_defaults('image_path'), address_port=YOLO_CLIENT.get_defaults('address_port'),
            prediction_path=YOLO_CLIENT.get_defaults('prediction_path'), signature=YOLO_CLIENT.get_defaults('signature')):
    # load the input image and construct the payload for the request
    image = open(image_path, "rb").read()
    payload = {"image": image}

    # submit the request
    keras_url_api = 'http://'+address_port+'/'+signature
    r = requests.post(keras_url_api, files=payload).json()

    # ensure the request was successful
    filename = os.path.basename(image_path)
    filelist = []
    image_raw = Image.open(image_path)

    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    if r["success"]:
        # loop over the predictions and display them
        for (i, result) in enumerate(r["item"]):
            predicted_class, score, left, top, right, bottom = result.split(',')
            print(predicted_class, score, left, top, right, bottom)
            filename_precess = '{}_{}_{}'.format(predicted_class, i, filename)

            cropbox = (int(left), int(top), int(right), int(bottom))
            cropped = image_raw.crop(cropbox)
            prediction_image = os.path.join(prediction_path, filename_precess)
            cropped.save(prediction_image)
            print('Saving: ', prediction_image)
            filelist.append(prediction_image)

    # otherwise, the request failed
    else:
        print("Request failed")

    return filelist


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--address_port', type=str, default=YOLO_CLIENT.get_defaults('address_port'),
        help='service address and port, default: 192.168.33.55:5000'
    )

    parser.add_argument(
        '--signature', type=str, default=YOLO_CLIENT.get_defaults('signature'),
        help='signature, default: predict'
    )

    parser.add_argument(
        '--prediction_path', type=str, default=YOLO_CLIENT.get_defaults('prediction_path'),
        help='path to prediction directory'
    )

    parser.add_argument(
        '--image_path', type=str, default=YOLO_CLIENT.get_defaults('image_path'),
        help='path to image'
    )


    FLAGS, unparsed = parser.parse_known_args()

    response_list = yolo_request(FLAGS.image_path, FLAGS.address_port, FLAGS.prediction_path, FLAGS.signature)
    print(response_list)

