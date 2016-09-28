# !/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --server=localhost:9000
"""

import sys
import threading

# This is a placeholder for a Google-internal import.
import grpc
from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.example import mnist_input_data
from flask import Flask


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS
app = Flask(__name__)

hostport = "localhost:9000"
work_dir = "/tmp"
concurrency = 1


class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1


def _create_rpc_callback(label, result_counter):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print exception
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['scores'].float_val)
      prediction = numpy.argmax(response)
      if label != prediction:
        result_counter.inc_error()
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


@app.route("/")
def do_inference():
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  test_data_set = mnist_input_data.read_data_sets(work_dir).test
  host, port = hostport.split(':')
  creds = grpc.ssl_channel_credentials(open('roots.pem').read())
  channel = implementations.secure_channel(host, int(port), creds)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = _ResultCounter(1, concurrency)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'mnist'
  image, label = test_data_set.next_batch(1)
  request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
  result_counter.throttle()
  result_future = stub.Predict.future(request, 5.0)  # 5 seconds
  result_future.add_done_callback(
      _create_rpc_callback(label[0], result_counter))
  return result_counter.get_error_rate()


def main(_):
  app.run(host='0.0.0.0', port=8888)


if __name__ == '__main__':
  tf.app.run()
  app.run(host='0.0.0.0', port=8888)
