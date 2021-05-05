import time

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import normalize
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import norm, multivariate_normal
#import synthetic
import os
import json


def run_clustering(filepath=None, savepath=None, shorten=False):
    if filepath is None:
        filepath = "../evaluation"
    if savepath is None:
        savepath = "clustering_results.json"
    names, observations = load_features(filepath)

    observations = normalize(observations, axis=0, norm='max')
    if shorten:
        observations = observations[:100, :]
        names = names[:100]
    results = cluster(observations=observations)
    print(np.shape(observations))
    print(np.shape(names))
    print(np.shape(results))
    combined = np.concatenate((names, results), axis=0)
    lists = combined.tolist()
    with open(savepath, 'w') as outfile:
        json.dump(lists, outfile)




from tensorflow.python.keras.backend import set_session
def session_options(enable_gpu_ram_resizing=True):
  """Convenience function which sets common `tf.Session` options."""
  config = tf.ConfigProto()
  config.log_device_placement = True
  if enable_gpu_ram_resizing:
    config.gpu_options.allow_growth = True
  return config

def reset_sess(config=None):
  """Convenience function to create the TF graph and session, or reset them."""
  if config is None:
    config = session_options()
  tf.reset_default_graph()
  global sess
  try:
    sess.close()
  except:
    pass
  sess = tf.InteractiveSession(config=config)


def cluster(observations):
    num_samples, dims = np.shape(observations)
    tfd = tfp.distributions
    reset_sess()
    # set_session(sess)
    # Upperbound on K
    max_cluster_num = 60

    dtype = np.float64
    # Define trainable variables.
    mix_probs = tf.nn.softmax(
        tf.Variable(
            name='mix_probs',
            initial_value=np.ones([max_cluster_num], dtype) / max_cluster_num))

    loc = tf.Variable(
        name='loc',
        initial_value=np.random.uniform(
            low=0,  #
            high=1,  # set around maximum value of sample value
            size=[max_cluster_num, dims]))

    precision = tf.nn.softplus(tf.Variable(
        name='precision',
        initial_value=
        np.ones([max_cluster_num, dims], dtype=dtype)))

    alpha = tf.nn.softplus(tf.Variable(
        name='alpha',
        initial_value=
        np.ones([1], dtype=dtype)))

    training_vals = [mix_probs, alpha, loc, precision]

    # Prior distributions of the training variables

    # Use symmetric Dirichlet prior as finite approximation of Dirichlet process.
    rv_symmetric_dirichlet_process = tfd.Dirichlet(
        concentration=np.ones(max_cluster_num, dtype) * alpha / max_cluster_num,
        name='rv_sdp')

    rv_loc = tfd.Independent(
        tfd.Normal(
            loc=tf.zeros([max_cluster_num, dims], dtype=dtype),
            scale=tf.ones([max_cluster_num, dims], dtype=dtype)),
        reinterpreted_batch_ndims=1,
        name='rv_loc')

    rv_precision = tfd.Independent(
        tfd.InverseGamma(
            concentration=np.ones([max_cluster_num, dims], dtype),
            rate=np.ones([max_cluster_num, dims], dtype)),
        reinterpreted_batch_ndims=1,
        name='rv_precision')

    rv_alpha = tfd.InverseGamma(
        concentration=np.ones([1], dtype=dtype),
        rate=np.ones([1]),
        name='rv_alpha')

    # Define mixture model
    rv_observations = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=mix_probs),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc,
            scale_diag=precision))

    # Learning rates and decay
    starter_learning_rate = 1e-6
    end_learning_rate = 1e-10
    decay_steps = 1e4

    # Number of training steps
    training_steps = 500

    # Mini-batch size
    batch_size = 20

    # Sample size for parameter posteriors
    sample_size = 100

    # Placeholder for mini-batch
    # observations_tens = tf.compat.v1.placeholder(dtype, shape=[batch_size, dims], name='observations_ten') # this is the bad boi, not fed a value
    observations_tens = tf.placeholder(dtype, shape=[batch_size, dims], name='observations_ten')
    print(observations_tens)
    print(dtype)
    # Define joint log probabilities
    # Notice that each prior probability should be divided by num_samples and
    # likelihood is divided by batch_size for pSGLD optimization.
    log_prob_parts = [
        rv_loc.log_prob(loc) / num_samples,
        rv_precision.log_prob(precision) / num_samples,
        rv_alpha.log_prob(alpha) / num_samples,
        rv_symmetric_dirichlet_process.log_prob(mix_probs)[..., tf.newaxis]
        / num_samples,
        rv_observations.log_prob(observations_tens) / batch_size
    ]
    tfvar_log_prob_parts = []
    varnames = ['rvloc', 'rvprecision', 'rvalpha', 'rvdirichlet', 'rvobservations']
    for e, v in enumerate(log_prob_parts):  # convert log prob into variables
        tfvar_log_prob_parts.append(tf.Variable(initial_value=v, name=varnames[e]))

    joint_log_prob = tf.reduce_sum(tf.concat(log_prob_parts, axis=-1), axis=-1)
    # print(joint_log_prob)
    # Make mini-batch generator
    dx = tf.data.Dataset.from_tensor_slices(observations) \
        .shuffle(500).repeat().batch(batch_size)
    # iterator = tf.compat.v1.data.make_one_shot_iterator(dx)
    iterator = tf.data.make_one_shot_iterator(dx)
    next_batch = iterator.get_next()

    # Define learning rate scheduling
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.polynomial_decay(
        starter_learning_rate,
        global_step, decay_steps,
        end_learning_rate, power=1.)

    # Set up the optimizer. Don't forget to set data_size=num_samples.
    optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(
        learning_rate=learning_rate,
        preconditioner_decay_rate=0.99,
        burnin=1500,
        data_size=num_samples)

    # train_op = optimizer_kernel.minimize(-joint_log_prob, var_list=tuple(log_prob_parts), tape=tf.GradientTape(persistent=False)) # guess by daniel

    # def joint_log_prob():
    #  val = tf.reduce_sum(tf.concat(tfvar_log_prob_parts, axis=-1), axis=-1)
    #  return val

    # init = tf.global_variables_initializer()
    # sess.run(init, feed_dict={observations_tens: np.zeros(shape=[batch_size, dims])})

    # train_op = optimizer_kernel.minimize(joint_log_prob, var_list = tfvar_log_prob_parts )
    train_op = optimizer_kernel.minimize(joint_log_prob)
    # Arrays to store samples
    mean_mix_probs_mtx = np.zeros([training_steps, max_cluster_num])
    mean_alpha_mtx = np.zeros([training_steps, 1])
    mean_loc_mtx = np.zeros([training_steps, max_cluster_num, dims])
    mean_precision_mtx = np.zeros([training_steps, max_cluster_num, dims])

    init = tf.global_variables_initializer()
    sess.run(init, feed_dict={observations_tens: np.zeros(shape=[batch_size, dims], dtype=np.double)})

    start = time.time()
    for it in range(training_steps):
        [mean_mix_probs_mtx[it, :], mean_alpha_mtx[it, 0], mean_loc_mtx[it, :, :], mean_precision_mtx[it, :, :],
         _] = sess.run([
            *training_vals,
            train_op
        ], feed_dict={
            observations_tens: sess.run(next_batch)})

    elapsed_time_psgld = time.time() - start
    print("Elapsed time: {} seconds".format(elapsed_time_psgld))

    # Take mean over the last sample_size iterations
    mean_mix_probs_ = mean_mix_probs_mtx[-sample_size:, :].mean(axis=0)
    mean_alpha_ = mean_alpha_mtx[-sample_size:, :].mean(axis=0)
    mean_loc_ = mean_loc_mtx[-sample_size:, :].mean(axis=0)
    mean_precision_ = mean_precision_mtx[-sample_size:, :].mean(axis=0)

    posterior_loc = tf.placeholder(
        dtype, [None, max_cluster_num, dims], name='posterior_loc')
    posterior_precision = tf.placeholder(
        dtype, [None, max_cluster_num, dims], name='posterior_precision')
    posterior_probmix = tf.placeholder(
        dtype, [None, max_cluster_num], name='posterior_probmix')

    # Posterior of z (unnormalized)
    uposterior = tfd.MultivariateNormalDiag(loc=posterior_loc, scale_diag=posterior_precision) \
                     .log_prob(tf.expand_dims(tf.expand_dims(observations, axis=1), axis=1)) \
                 + tf.log(posterior_probmix[tf.newaxis, ...])

    # normalize posterior of z over the latent states
    posterior = uposterior - tf.reduce_logsumexp(uposterior, axis=-1)[..., tf.newaxis]

    cluster_asgmt = sess.run(tf.argmax(
        tf.reduce_mean(posterior, axis=1), axis=1), feed_dict={
        posterior_loc: mean_loc_mtx[-sample_size:, :],
        posterior_precision: mean_precision_mtx[-sample_size:, :],
        posterior_probmix: mean_mix_probs_mtx[-sample_size:, :]})

    idxs, count = np.unique(cluster_asgmt, return_counts=True)
    return cluster_asgmt
    #for e, i in enumerate(count):
    #    print(f'cluster {e} has {i} elements \n')

    def convert_int_elements_to_consecutive_numbers_in(array):
        unique_int_elements = np.unique(array)
        for consecutive_number, unique_int_element in enumerate(unique_int_elements):
            array[array == unique_int_element] = consecutive_number
        return array

def float_array(arr):
	return np.array(arr, dtype=np.float64)

def load_features(dir_path: str): #-> Tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(dir_path, 'features.json'), 'r') as f:
        data = json.load(f)
        keys = np.array([k for k in data])
        features = float_array([data[k] for k in keys])
        return keys, features

    #cmap = plt.get_cmap('tab10')
    #plt.scatter(
    #    observations[:, 0], observations[:, 1],
    #    1,
    #    c=cmap(convert_int_elements_to_consecutive_numbers_in(cluster_asgmt)))
    #plt.axis([0, 1, 0, 1])
    #plt.xlabel("tempo")
    #plt.ylabel("loudness")
    #plt.show()


if __name__ == '__main__':
    run_clustering('C:/Users/danie/OneDrive/Documents/GitHub/cfm/backend/evaluation/data/', 'C:/Users/danie/OneDrive/Documents/GitHub/cfm/backend/evaluation/clustering_results.json', shorten=False)
