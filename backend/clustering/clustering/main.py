import attr
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
from numbers import Real
from tensorflow_probability import distributions as tfd

Mixture = tfd.MixtureSameFamily
Cat = tfd.Categorical
NormalDiag = tfd.MultivariateNormalDiag
Dir = tfd.Dirichlet
Ind = tfd.Independent
Normal = tfd.Normal
InvGamma = tfd.InverseGamma


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

	def run(self, data):
		dtype = np.float64
		num_samples, dims = np.shape(data)
		one = np.ones([1], dtype=dtype)
		ones_vector = np.ones([self.max_clusters], dtype=dtype)
		ones_matrix = np.ones([self.max_clusters, dims], dtype=dtype)
		zeros_matrix = np.zeros([self.max_clusters, dims], dtype=dtype)
		# Define trainable variables.
		mix_probs = tf.nn.softmax(
			tf.Variable(
				name='mix_probs',
				initial_value=ones_vector / self.max_clusters))
		loc = tf.Variable(
			name='loc',
			initial_value=np.random.uniform(
				low=0, high=1, size=[self.max_clusters, dims]))
		precision = tf.Variable(name='precision', initial_value=ones_matrix)
		precision = tf.nn.softplus(precision)
		alpha = tf.nn.softplus(tf.Variable(name='alpha', initial_value=one))
		training_vals = [mix_probs, alpha, loc, precision]
		# Prior distributions of the training variables
		# Use symmetric Dirichlet prior as finite approximation of Dirichlet
		# process.
		concentration = ones_vector * alpha / self.max_clusters
		rv_sym_dir_proc = Dir(concentration=concentration, name='rv_sdp')
		rv_loc = Ind(
			Normal(loc=zeros_matrix, scale=ones_matrix),
			reinterpreted_batch_ndims=1,
			name='rv_loc')
		rv_precision = Ind(
			InvGamma(concentration=ones_matrix, rate=ones_matrix),
			reinterpreted_batch_ndims=1,
			name='rv_precision')
		rv_alpha = InvGamma(concentration=one, rate=one, name='rv_alpha')
		# Define mixture model
		rv_data = Mixture(
			mixture_distribution=Cat(probs=mix_probs),
			components_distribution=NormalDiag(loc=loc, scale_diag=precision))
		data_ten = tf.placeholder(
			dtype, shape=[self.batch_size, dims], name='data_ten')
		# Define joint log probabilities
		# Notice that each prior probability should be divided by num_samples
		# and
		# likelihood is divided by batch_size for pSGLD optimization.
		log_prob_parts = [
			rv_loc.log_prob(loc) / num_samples,
			rv_precision.log_prob(precision) / num_samples,
			rv_alpha.log_prob(alpha) / num_samples,
			rv_sym_dir_proc.log_prob(mix_probs)[..., tf.newaxis] / num_samples,
			rv_data.log_prob(data_ten) / self.batch_size]
		var_log_prob_parts = []
		var_names = ['rv_loc', 'rv_precision', 'rv_alpha', 'rv_dir', 'rv_data']
		for e, v in enumerate(
				log_prob_parts):  # convert log prob into variables
			var_log_prob_parts.append(
				tf.Variable(initial_value=v, name=var_names[e]))
		joint_log_prob = tf.concat(log_prob_parts, axis=-1)
		joint_log_prob = tf.reduce_sum(joint_log_prob, axis=-1)
		# Make mini-batch generator
		dx = tf.data.Dataset.from_tensor_slices(data)
		dx = dx.shuffle(self.buffer_size).repeat().batch(self.batch_size)
		iterator = tf.data.make_one_shot_iterator(dx)
		next_batch = iterator.get_next()
		# Define learning rate scheduling
		global_step = tf.Variable(0, trainable=False, name='global_step')
		lr = tf.train.polynomial_decay(
			self.start_lr, global_step, self.decay_steps, self.end_lr, power=1)
		# Set up the optimizer. Don't forget to set data_size=num_samples.
		optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(
			learning_rate=lr,
			preconditioner_decay_rate=0.99,
			burnin=1500,
			data_size=num_samples)
		train_op = optimizer_kernel.minimize(joint_log_prob)
		# Arrays to store samples
		mean_mix_probs = np.zeros([self.train_steps, self.max_clusters])
		mean_alpha = np.zeros([self.train_steps, 1])
		mean_location = np.zeros([self.train_steps, self.max_clusters, dims])
		mean_precision = np.zeros([self.train_steps, self.max_clusters, dims])
		init = tf.global_variables_initializer()
		feed = {data_ten: np.zeros([self.batch_size, dims], dtype=np.double)}
		sess.run(init, feed_dict=feed)
		start = time.time()
		for it in range(self.train_steps):
			feed = {data_ten: sess.run(next_batch)}
			result = sess.run([*training_vals, train_op], feed_dict=feed)
			mean_mix_probs[it, :], mean_alpha[it, 0] = result[:1]
			mean_location[it, :, :], mean_precision[it, :, :] = result[1:-1]
		elapsed_time_psgld = time.time() - start
		print("Elapsed time: {} seconds".format(elapsed_time_psgld))
		posterior_loc = tf.placeholder(
			dtype, [None, self.max_clusters, dims], name='posterior_loc')
		post_prec = tf.placeholder(
			dtype, [None, self.max_clusters, dims], name='posterior_precision')
		posterior_probmix = tf.placeholder(
			dtype, [None, self.max_clusters], name='posterior_probmix')
		# Posterior of z (un-normalized)
		posterior = NormalDiag(loc=posterior_loc, scale_diag=post_prec)
		posterior = posterior.log_prob(
			tf.expand_dims(tf.expand_dims(data, axis=1), axis=1))
		posterior += tf.math.log(posterior_probmix[tf.newaxis, ...])
		# normalize posterior of z over the latent states
		log_sum_exp = tf.reduce_logsumexp(posterior, axis=-1)[..., tf.newaxis]
		posterior = posterior - log_sum_exp
		clusters = sess.run(tf.argmax(
			tf.reduce_mean(posterior, axis=1), axis=1), feed_dict={
			posterior_loc: mean_location[-self.sample_size:, :],
			post_prec: mean_precision[-self.sample_size:, :],
			posterior_probmix: mean_mix_probs[-self.sample_size:, :]})
		return clusters
