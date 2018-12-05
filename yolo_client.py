from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from yolo3.utils import letterbox_image
from PIL import Image
import numpy as np
import grpc

hostport = '192.168.33.55:8500'

img_path = '5.jpg'


def image_precess(image):
    boxed_image = letterbox_image(image, tuple(reversed((416, 416))))

    image_data = np.array(boxed_image, dtype='float32')
    image_data = np.multiply(image_data, 1.0 / 255.0)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    # image_data = image_data.astype(np.float32)

    return image_data


def _create_rpc_callback():
    def _callback(result_future):
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            print(result_future.result().outputs['outputs'])

    return _callback


if __name__ == '__main__':

    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    image = Image.open(img_path)
    data = image_precess(image)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolo'
    request.model_spec.signature_name = 'classify'
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data)
    )
    result_future = stub.Predict(request, 5.0) # 10 secs timeout
    print(result_future)
    # result_future.add_done_callback(_create_rpc_callback())

