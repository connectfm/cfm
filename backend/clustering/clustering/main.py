import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tensorflow_probability import distributions as tfd


def session_options(enable_gpu_ram_resizing=True):
	"""Convenience function which sets common `tf.Session` options."""
	config = tf.ConfigProto()
	config.log_device_placement = True
	if enable_gpu_ram_resizing:
		config.gpu_options.allow_growth = True
	return config


def reset_sess(config=None):
	"""Convenience function to create the TF graph and session, or reset."""
	if config is None:
		config = session_options()
	tf.reset_default_graph()
	global sess
	try:
		sess.close()
	except:
		pass
	sess = tf.InteractiveSession(config=config)


def cluster(data):
	num_samples, dims = np.shape(data)
	reset_sess()
	set_session(sess)
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
	# Use symmetric Dirichlet prior as finite approximation of Dirichlet
	# process.
	concentration = np.ones(max_cluster_num, dtype) * alpha / max_cluster_num
	rv_sym_dir_proc = tfd.Dirichlet(
		concentration=concentration, name='rv_sdp')
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
	rv_data = tfd.MixtureSameFamily(
		mixture_distribution=tfd.Categorical(probs=mix_probs),
		components_distribution=tfd.MultivariateNormalDiag(
			loc=loc, scale_diag=precision))
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
	data_ten = tf.placeholder(dtype, shape=[batch_size, dims], name='data_ten')
	print(data_ten)
	print(dtype)
	# Define joint log probabilities
	# Notice that each prior probability should be divided by num_samples and
	# likelihood is divided by batch_size for pSGLD optimization.
	log_prob_parts = [
		rv_loc.log_prob(loc) / num_samples,
		rv_precision.log_prob(precision) / num_samples,
		rv_alpha.log_prob(alpha) / num_samples,
		rv_sym_dir_proc.log_prob(mix_probs)[..., tf.newaxis] / num_samples,
		rv_data.log_prob(data_ten) / batch_size]
	var_log_prob_parts = []
	var_names = ['rv_loc', 'rv_precision', 'rv_alpha', 'rv_dir', 'rv_data']
	for e, v in enumerate(log_prob_parts):  # convert log prob into variables
		var_log_prob_parts.append(
			tf.Variable(initial_value=v, name=var_names[e]))
	joint_log_prob = tf.reduce_sum(tf.concat(log_prob_parts, axis=-1), axis=-1)
	# Make mini-batch generator
	dx = tf.data.Dataset.from_tensor_slices(data)
	dx = dx.shuffle(500).repeat().batch(batch_size)
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
	train_op = optimizer_kernel.minimize(joint_log_prob)
	# Arrays to store samples
	mean_mix_probs_mtx = np.zeros([training_steps, max_cluster_num])
	mean_alpha_mtx = np.zeros([training_steps, 1])
	mean_loc_mtx = np.zeros([training_steps, max_cluster_num, dims])
	mean_precision_mtx = np.zeros([training_steps, max_cluster_num, dims])
	init = tf.global_variables_initializer()
	feed = {data_ten: np.zeros(shape=[batch_size, dims], dtype=np.double)}
	sess.run(init, feed_dict=feed)
	start = time.time()
	for it in range(training_steps):
		feed = {data_ten: sess.run(next_batch)}
		result = sess.run([*training_vals, train_op], feed_dict=feed)
		mean_mix_probs_mtx[it, :], mean_alpha_mtx[it, 0] = result[:1]
		mean_loc_mtx[it, :, :], mean_precision_mtx[it, :, :] = result[1:-1]
	elapsed_time_psgld = time.time() - start
	print("Elapsed time: {} seconds".format(elapsed_time_psgld))
	posterior_loc = tf.placeholder(
		dtype, [None, max_cluster_num, dims], name='posterior_loc')
	posterior_precision = tf.placeholder(
		dtype, [None, max_cluster_num, dims], name='posterior_precision')
	posterior_probmix = tf.placeholder(
		dtype, [None, max_cluster_num], name='posterior_probmix')
	# Posterior of z (un-normalized)
	posterior = tfd.MultivariateNormalDiag(
		loc=posterior_loc, scale_diag=posterior_precision)
	posterior = posterior.log_prob(
		tf.expand_dims(tf.expand_dims(data, axis=1), axis=1))
	posterior += tf.log(posterior_probmix[tf.newaxis, ...])
	# normalize posterior of z over the latent states
	log_sum_exp = tf.reduce_logsumexp(posterior, axis=-1)[..., tf.newaxis]
	posterior = posterior - log_sum_exp
	clusters = sess.run(tf.argmax(
		tf.reduce_mean(posterior, axis=1), axis=1), feed_dict={
		posterior_loc: mean_loc_mtx[-sample_size:, :],
		posterior_precision: mean_precision_mtx[-sample_size:, :],
		posterior_probmix: mean_mix_probs_mtx[-sample_size:, :]})
	return clusters
