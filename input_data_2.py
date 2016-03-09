# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import tensorflow.python.platform
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def dense_to_one_hot(labels_dense, num_classes=4):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

class DataSet(object):
  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      pass
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)

      # Now assume depth == 3

      assert images.shape[3] == 3
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2] * 3)
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      pass
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
def read_data_sets(num_training=6360, num_val=400, num_test=400, fake_data=False, one_hot=False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if fake_data:
    pass

  import csv
  import dateutil.parser
  from scipy.misc import imread
  import numpy as np
  import random
  from shutil import copyfile

  NUM_BUCKETS = 4

  BUCKET_DELTA = 24*60*60 / NUM_BUCKETS

  DATASET_DIRECTORY = 'time_capsule/'

  X = []
  y = []

  capsules = ['morning', 'afternoon', 'evening', 'night']

  image_id_set = set()
  for capsule_id, capsule in enumerate(capsules):
    f = open(DATASET_DIRECTORY + capsule + '/' + capsule + '.csv', 'rU')
    reader = csv.reader(f)
    for row in reader:
      # Extract image metadata from CSV row
      image_id = row[0]
      date_str = row[1]
      url = row[2]

      # Exclude duplicate images
      if image_id in image_id_set:
        continue
      image_id_set.add(image_id)

      # Determine which bucket the image is in
      try:
        date_obj = dateutil.parser.parse(date_str)
      except:
        continue
      seconds = date_obj.hour * 60 * 60 + date_obj.minute * 60 + date_obj.second
      bucket_id = None
      for i in range(NUM_BUCKETS):
        if seconds < (i + 1) * BUCKET_DELTA:
          bucket_id = i
          break

      # Load the image
      try:
        # read images from the directory
        filename = DATASET_DIRECTORY + capsule + '/d/' + image_id + '.png'
        img = imread(filename)

        # This code allows us to visualize what images are in each bucket
        # dst = 'cs231n/datasets/time_capsule_buckets/' + str(bucket_id) + '/' + image_id + '.png'
        # copyfile(filename, dst)
      except:
        continue
      X_block = img.transpose(2, 0, 1)
      s = X_block.shape
      if s[0] == 3 and s[1] == 150 and s[2] == 150: # check if dimensions are correct
        X.append(X_block)
        #y.append(bucket_id)
        y.append(capsule_id)
  f.close()

  # Number of images in the dataset
  num_images = len(X)

  # Shuffle the dataset
  mask = range(num_images)
  random.seed(1234)
  random.shuffle(mask)
  X = [X[i] for i in mask]
  y = [y[i] for i in mask]

  # Convert X and Y to ndarray
  tempX = np.zeros((num_images, 3, 150, 150))
  for i in range(num_images):
      tempX[i, :, :, :] = X[i]
  X = tempX
  y = np.array(y, dtype=np.int32)
  print("Shapes: ", X.shape, y.shape)
  # Create training, validation, and test sets

  # If not enough images, augment the training set by flipping images horizontally
  num_training_start = num_images - num_val - num_test
  if num_training > num_training_start:
    num_training_more = num_training - num_training_start
    X_train = np.zeros((num_training, 3, 150, 150))
    y_train = np.zeros((num_training), dtype=np.int32)
    X_train[0:num_training_start, :, :, :] = X[0:num_training_start, :, :, :]
    y_train[0:num_training_start] = y[0:num_training_start]
    for i in range(num_training_more):
      X_train[num_training_start + i, :, :, :] = X[i, :, :, ::-1]
      y_train[num_training_start + i] = y[i]
  else:
    X_train = X[0:num_training, :, :, :]
    y_train = y[0:num_training]

  X_val = X[num_training_start:(num_training_start+num_val), :, :, :]
  y_val = y[num_training_start:(num_training_start+num_val)]

  X_test = X[(num_training_start+num_val):(num_training_start+num_val+num_test), :, :, :]
  y_test = y[(num_training_start+num_val):(num_training_start+num_val+num_test)]


  # Reshape to (N, H, W, C)
  # e.g. (6360, 3, 150, 150) -> (6360, 150, 150, 3)
  X_train = np.transpose(X_train, (0, 2, 3, 1))
  X_val = np.transpose(X_val, (0, 2, 3, 1))
  X_test = np.transpose(X_test, (0, 2, 3, 1))


  train_images = X_train#tf.convert_to_tensor(X_train)

  if one_hot:
    train_labels = dense_to_one_hot(y_train)
  else:
    train_labels = y_train

  test_images = X_test#tf.convert_to_tensor(X_test)

  if one_hot:
    test_labels = dense_to_one_hot(y_test)
  else:
    test_labels = y_test

  validation_images = X_val#tf.convert_to_tensor(X_val)

  if one_hot:
    validation_labels = dense_to_one_hot(y_val)
  else:
    validation_labels = y_val

  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, dtype=dtype)
  return data_sets
