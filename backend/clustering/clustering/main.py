import attr
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numbers import Real
from tensorflow_probability import distributions as tfd
from typing import Any

Mixture = tfd.MixtureSameFamily
Cat = tfd.Categorical
NormalDiag = tfd.MultivariateNormalDiag
Dir = tfd.Dirichlet
Ind = tfd.Independent
Normal = tfd.Normal
InvGamma = tfd.InverseGamma

DTYPE = np.float64


@attr.s(slots=True)
class DirichletProcessMixture:
	start_lr = attr.ib(type=Real, default=1e-6)
	end_lr = attr.ib(type=Real, default=1e-10)
	decay_steps = attr.ib(type=int, default=10_000)
	train_steps = attr.ib(type=int, default=500)
	# Mini-batch size
	batch_size = attr.ib(type=int, default=20)
	# Sample size for parameter posteriors
	sample_size = attr.ib(type=int, default=100)
	max_clusters = attr.ib(type=int, default=60)
	buffer_size = attr.ib(type=int, default=500)
	seed = attr.ib(type=Any, default=None)
	samples = attr.ib(type=int, init=False)
	dims = attr.ib(type=int, init=False)
	_one = attr.ib(type=np.ndarray, init=False, repr=False)
	_ones_vec = attr.ib(type=np.ndarray, init=False, repr=False)
	_ones_mat = attr.ib(type=np.ndarray, init=False, repr=False)
	_zeros_mat = attr.ib(type=np.ndarray, init=False, repr=False)
	_mix_probs = attr.ib(type=np.ndarray, init=False, repr=False)
	_alpha = attr.ib(type=np.ndarray, init=False, repr=False)
	_loc = attr.ib(type=np.ndarray, init=False, repr=False)
	_precision = attr.ib(type=np.ndarray, init=False, repr=False)

	def __attrs_post_init__(self):
		self._rng = np.random.default_rng(self.seed)

	def initialize(self, data: np.ndarray):
		n_samples, n_dims = np.shape(data)
		self.samples, self.dims = n_samples, n_dims
		self._one = np.ones([1], dtype=DTYPE)
		self._ones_vec = np.ones((self.max_clusters,), dtype=DTYPE)
		self._ones_mat = np.ones((self.max_clusters, n_dims), dtype=DTYPE)
		self._zeros_mat = np.zeros_like(self._ones_mat)
		self._mix_probs = np.zeros((self.train_steps, self.max_clusters))
		self._alpha = np.zeros((self.train_steps, 1))
		self._loc = np.zeros((self.train_steps, self.max_clusters, self.dims))
		self._precision = np.copy(self._loc)

	def run(self, data):
		# Prior distributions of the training variables
		# Use symmetric Dirichlet prior as finite approximation of Dirichlet
		# process.
		data_ten = tf.placeholder(
			DTYPE, shape=[self.batch_size, self.dims], name='data_ten')
		training_vars = self._training_variables()
		mix_probs, alpha, loc, precision = training_vars
		components = NormalDiag(loc=loc, scale_diag=precision)
		mixture = Mixture(Cat(probs=mix_probs), components)
		joint_log_prob = self._joint_log_prob(training_vars, data_ten, mixture)
		# Make mini-batch generator
		dx = tf.data.Dataset.from_tensor_slices(data)
		dx = dx.shuffle(self.buffer_size).repeat().batch(self.batch_size)
		iterator = tf.data.make_one_shot_iterator(dx)
		next_batch = iterator.get_next()
		# Define learning rate scheduling
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.polynomial_decay(
			self.start_lr, global_step, self.decay_steps, self.end_lr, power=1)
		optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(
			learning_rate=lr,
			preconditioner_decay_rate=0.99,
			burnin=1500,
			data_size=self.samples)
		train_op = optimizer_kernel.minimize(joint_log_prob)
		# Arrays to store samples
		init = tf.global_variables_initializer()
		feed = {
			data_ten: np.zeros([self.batch_size, self.dims],
							   dtype=np.double)}
		sess.run(init, feed_dict=feed)
		for i in range(self.train_steps):
			feed = {data_ten: sess.run(next_batch)}
			result = sess.run((*training_vars, train_op), feed_dict=feed)
			self._mix_probs[i, :], self._alpha[i, 0] = result[:2]
			self._loc[i, :, :], self._precision[i, :, :] = result[2:]
		post_loc = tf.placeholder(
			DTYPE, [None, self.max_clusters, self.dims],
			name='posterior_loc')
		post_prec = tf.placeholder(
			DTYPE, [None, self.max_clusters, self.dims],
			name='posterior_precision')
		post_mix = tf.placeholder(
			DTYPE, [None, self.max_clusters], name='posterior_probmix')
		# Posterior of z (un-normalized)
		reshaped = tf.expand_dims(tf.expand_dims(data, axis=1), axis=1)
		posterior = NormalDiag(loc=post_loc, scale_diag=post_prec)
		posterior = posterior.log_prob(reshaped)
		posterior += tf.math.log(post_mix[tf.newaxis, ...])
		# normalize posterior of z over the latent states
		log_sum_exp = tf.reduce_logsumexp(posterior, axis=-1)[..., tf.newaxis]
		posterior = posterior - log_sum_exp
		clusters = sess.run(tf.argmax(
			tf.reduce_mean(posterior, axis=1), axis=1), feed_dict={
			post_loc: self._loc[-self.sample_size:, :],
			post_prec: self._precision[-self.sample_size:, :],
			post_mix: self._mix_probs[-self.sample_size:, :]})
		return clusters

	def _training_variables(self):
		mix_probs = tf.Variable(self._ones_vec / self.max_clusters)
		mix_probs = tf.nn.softmax(mix_probs)
		alpha = tf.nn.softplus(tf.Variable(self._one))
		loc = self._rng.uniform(size=(self.max_clusters, self.dims))
		loc = tf.Variable(loc)
		precision = tf.nn.softplus(tf.Variable(self._ones_mat))
		return mix_probs, alpha, loc, precision

	def _joint_log_prob(self, training_vars, data_ten, mixture):
		mix_probs, alpha, loc, precision = training_vars
		loc = Normal(self._zeros_mat, self._ones_mat)
		loc = Ind(loc, reinterpreted_batch_ndims=1)
		precision = InvGamma(self._ones_mat, self._ones_mat)
		precision = Ind(precision, reinterpreted_batch_ndims=1)
		alpha = InvGamma(concentration=self._one, rate=self._one)
		dirichlet = Dir(self._ones_vec * alpha / self.max_clusters)
		log_probs = (
			loc.log_prob(loc) / self.samples,
			precision.log_prob(precision) / self.samples,
			alpha.log_prob(alpha) / self.samples,
			dirichlet.log_prob(mix_probs)[..., tf.newaxis] / self.samples,
			mixture.log_prob(data_ten) / self.batch_size)
		return tf.reduce_sum(tf.concat(log_probs, axis=-1), axis=-1)
