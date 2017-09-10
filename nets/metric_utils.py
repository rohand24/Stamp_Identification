from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import tensor_util
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.metrics.python.ops import confusion_matrix_ops
from tensorflow.contrib.metrics.python.ops import set_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables

def _safe_div(numerator, denominator, name):
  """Divides two values, returning 0 if the denominator is <= 0.

  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  return math_ops.select(
      math_ops.greater(denominator, 0),
      math_ops.truediv(numerator, denominator),
      0,
      name=name)


def _safe_scalar_div(numerator, denominator, name):
  """Divides two values, returning 0 if the denominator is 0.

  Args:
    numerator: A scalar `float64` `Tensor`.
    denominator: A scalar `float64` `Tensor`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` == 0, else `numerator` / `denominator`
  """
  numerator.get_shape().with_rank_at_most(1)
  denominator.get_shape().with_rank_at_most(1)
  return control_flow_ops.cond(
      math_ops.equal(
          array_ops.constant(0.0, dtype=dtypes.float64), denominator),
      lambda: array_ops.constant(0.0, dtype=dtypes.float64),
      lambda: math_ops.div(numerator, denominator),
      name=name)

def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=dtypes.float32):
  """Creates a new local variable.

  Args:
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    collections: A list of collection names to which the Variable will be added.
    validate_shape: Whether to validate the shape of the variable.
    dtype: Data type of the variables.

  Returns:
    The created variable.
  """
  # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
  collections = list(collections or [])
  collections += [ops.GraphKeys.LOCAL_VARIABLES]
  return variables.Variable(
      initial_value=array_ops.zeros(shape, dtype=dtype),
      name=name,
      trainable=False,
      collections=collections,
      validate_shape=validate_shape)

def streaming_mean(values, weights=None, metrics_collections=None,
                   updates_collections=None, name=None):
  """Computes the (weighted) mean of the given values.

  The `streaming_mean` function creates two local variables, `total` and `count`
  that are used to compute the average of `values`. This average is ultimately
  returned as `mean` which is an idempotent operation that simply divides
  `total` by `count`.

  For estimation of the metric  over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean`.
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights`, and it increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `Tensor` of arbitrary dimensions.
    weights: An optional `Tensor` whose shape is broadcastable to `values`.
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_scope name.

  Returns:
    mean: A tensor representing the current mean, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_value`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  with variable_scope.variable_scope(name, 'mean', [values, weights]):
    values = math_ops.to_float(values)

    total = _create_local('total', shape=[])
    count = _create_local('count', shape=[])

    if weights is not None:
      weights = math_ops.to_float(weights)
      values = math_ops.mul(values, weights)
      num_values = math_ops.reduce_sum(_broadcast_weights(weights, values))
    else:
      num_values = math_ops.to_float(array_ops.size(values))

    total_compute_op = state_ops.assign_add(total, math_ops.reduce_sum(values))
    count_compute_op = state_ops.assign_add(count, num_values)

    mean = _safe_div(total, count, 'value')
    with ops.control_dependencies([total_compute_op, count_compute_op]):
      update_op = _safe_div(total, count, 'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, mean)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean, update_op
    
def streaming_accuracy(predictions, labels, weights=None,
                       metrics_collections=None, updates_collections=None,
                       name=None):
  """Calculates how often `predictions` matches `labels`.

  The `streaming_accuracy` function creates two local variables, `total` and
  `count` that are used to compute the frequency with which `predictions`
  matches `labels`. This frequency is ultimately returned as `accuracy`: an
  idempotent operation that simply divides `total` by `count`.

  For estimation of the metric  over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `accuracy`.
  Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
  where the corresponding elements of `predictions` and `labels` match and 0.0
  otherwise. Then `update_op` increments `total` with the reduced sum of the
  product of `weights` and `is_correct`, and it increments `count` with the
  reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `Tensor` of any shape.
    labels: The ground truth values, a `Tensor` whose shape matches
      `predictions`.
    weights: An optional `Tensor` whose shape is broadcastable to `predictions`.
    metrics_collections: An optional list of collections that `accuracy` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    accuracy: A tensor representing the accuracy, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `accuracy`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions, labels, weights=weights)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  if labels.dtype != predictions.dtype:
    predictions = math_ops.cast(predictions, labels.dtype)
  is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
  return streaming_mean(is_correct, weights, metrics_collections,
                        updates_collections, name or 'accuracy')
                        
                        
                        
                        
def _remove_squeezable_dimensions(predictions, labels, weights):
  """Squeeze last dim if needed.

  Squeezes `predictions` and `labels` if their rank differs by 1.
  Squeezes `weights` if its rank is 1 more than the new rank of `predictions`

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    labels: Label values, a `Tensor` whose dimensions match `predictions`.
    weights: optional `weights` tensor. It will be squeezed if its rank is 1
      more than the new rank of `predictions`

  Returns:
    Tuple of `predictions`, `labels` and `weights`, possibly with the last
    dimension squeezed.
  """
  predictions, labels = tensor_util.remove_squeezable_dimensions(
      predictions, labels)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  if weights is not None:
    predictions_shape = predictions.get_shape()
    predictions_rank = predictions_shape.ndims
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims

    if (predictions_rank is not None) and (weights_rank is not None):
      # Use static rank.
      if weights_rank - predictions_rank == 1:
        weights = array_ops.squeeze(weights, [-1])
    elif (weights_rank is None) or (
        weights_shape.dims[-1].is_compatible_with(1)):
      # Use dynamic rank
      weights = control_flow_ops.cond(
          math_ops.equal(array_ops.rank(weights),
                         math_ops.add(array_ops.rank(predictions), 1)),
          lambda: array_ops.squeeze(weights, [-1]),
          lambda: weights)
  return predictions, labels, weights

                        