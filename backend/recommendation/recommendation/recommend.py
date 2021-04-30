import functools
import random
from typing import Any, Callable, Optional, Tuple

import attr
import codetiming
import numpy as np
from scipy.spatial import distance

import model
import util

DEFAULT_RATING = 2

logger = util.get_logger(__name__)


@attr.s(slots=True)
class Recommender:
	"""Recommendation system for connect.fm"""
	db = attr.ib(type=model.RecommendDB)
	metric = attr.ib(type=Callable, default=distance.euclidean)
	n_songs = attr.ib(type=int, default=None)
	cache = attr.ib(type=bool, default=False)
	seed = attr.ib(type=Any, default=None)
	_rng = attr.ib(type=np.random.Generator, init=False, repr=False)

	def __attrs_post_init__(self):
		random.seed(self.seed)
		self._rng = np.random.default_rng(self.seed)
		logger.debug(f'Distance metric: {self.metric}')
		logger.debug(f'Seed: {self.seed}')

	@codetiming.Timer(text='Time to recommend: {:0.4f} s', logger=logger.info)
	def recommend(self, name: str) -> str:
		"""Returns a recommended song based on a user."""
		logger.info(f'Retrieving a recommendation for user {name}')
		if (user := self.db.get_user(name)).taste is None:
			logger.warning(
				f'Unable to find the taste of user {user.name}. Using a taste '
				'from a random user')
			user.taste = self.db.get_random_taste()
		if len(neighbors := self.db.get_neighbors(user)) > 0:
			neighbors, tastes = self.db.get_features(*neighbors, song=False)
			if len(neighbors) > 0:
				ne = self.sample_neighbor(user, neighbors, tastes)
			else:
				logger.warning(
					'Unable to find any neighbors with a taste attribute. '
					f'Using user {user.name} as their own neighbor')
				ne = user
		else:
			logger.warning(
				f'Unable to find neighbors of user {user.name} either because '
				f'of missing attributes or because no users are present '
				f'within the set radius. Using user {user.name} as their own '
				f'neighbor')
			ne = user
		return self.sample_song(user, ne)

	def sample_neighbor(
			self,
			user: model.User,
			neighbors: np.ndarray,
			tastes: np.ndarray) -> model.User:
		"""Returns a neighbor using taste to weight the sampling."""
		logger.info(
			f'Sampling 1 of {len(neighbors)} neighbors of user {user.name}')
		similarity = util.similarity(user.taste, tastes, metric=self.metric)
		ne, idx = self.sample(neighbors, similarity, with_index=True)
		logger.info(f'Sampled neighbor {ne}')
		return model.User(ne, taste=tastes[idx])

	def sample(
			self,
			population: np.ndarray,
			weights: np.ndarray,
			*,
			with_index: bool = False) -> Any:
		"""Returns an element from the population using weighted sampling."""
		if (norm := sum(weights)) == 0:
			probs = np.zeros(weights.shape)
		else:
			probs = weights / norm
		mean = np.round(np.average(probs), 3)
		sd = np.round(np.std(probs), 3)
		logger.debug(f'Mean (sd) probability: {mean} ({sd})')
		if with_index:
			population = np.vstack((np.arange(len(population)), population)).T
			idx_and_chosen = self._rng.choice(population, p=probs)
			# String population converts indices to strings
			idx, chosen = int(idx_and_chosen[0]), idx_and_chosen[1:]
			chosen = chosen.item() if chosen.shape == (1,) else chosen
			logger.debug(f'Sampled element (index): {chosen} ({idx})')
			chosen = (chosen, idx)
		else:
			chosen = self._rng.choice(population, p=probs)
			logger.debug(f'Sampled element: {chosen}')
		return chosen

	def sample_song(self, user: model.User, ne: model.User) -> str:
		"""Returns a song based on user and neighbor contexts."""
		cluster = self.sample_cluster(user, ne)
		logger.info(f'Sampling a song to recommend')
		compute = functools.partial(
			lambda u, n: self.compute_ratings(u, n, cluster))
		song = self._sample(user, ne, compute)
		logger.info(f'Sampled song {song}')
		return song

	def _sample(
			self, user: model.User, ne: model.User, compute: Callable) -> Any:
		if cached := self.cache:
			cached = self.db.get_cached(user.name, ne.name, fuzzy=True)
		if cached:
			population, weights = cached
		else:
			population, weights = compute(user, ne)
		population, weights = np.array(population), util.float_array(weights)
		return self.sample(population, weights)

	def sample_cluster(self, user: model.User, ne: model.User) -> str:
		"""Returns a cluster based on user and neighbor contexts."""
		logger.info('Sampling a cluster from which to recommend a song')
		cluster = self._sample(user, ne, self.compute_scores)
		logger.info(f'Sampled cluster {cluster}')
		return cluster

	def compute_scores(
			self,
			user: model.User,
			ne: model.User) -> Tuple[np.ndarray, np.ndarray]:
		"""Computes cluster scores and caches the result"""
		logger.info(
			f'Computing cluster scores between user {user.name} and neighbor '
			f'{ne.name}')
		n_songs = self._get_num_songs()
		clusters, scores, per_cluster = [], [], []
		for i, cluster in enumerate(self.db.get_clusters()):
			clusters.append(cluster)
			scores.append(0)
			per_cluster.append(0)
			for song in self.db.get_songs(cluster, n_songs):
				scores[i] += self.adj_rating(user, ne, song)
				per_cluster[i] += 1
		# Normalizes based on the number of songs per cluster
		scores = [s / n for s, n in zip(scores, per_cluster)]
		logger.debug(f'Number of clusters: {len(clusters)}')
		logger.debug(f'Number of songs sampled: {sum(per_cluster)}')
		logger.debug(f'Number of songs sampled per cluster: {per_cluster}')
		if self.cache:
			self.db.cache((clusters, scores), user=user.name, ne=ne.name)
		return np.array(clusters), util.float_array(scores)

	def _get_num_songs(self) -> Optional[int]:
		if (n_songs := self.n_songs) is not None:
			if (n_clusters := self.db.get_num_clusters()) is None:
				logger.warning(
					f'Unable to find the number of clusters. Sampling '
					f'{n_songs} from each cluster')
			else:
				n_songs = int(np.ceil(self.n_songs / n_clusters))
				logger.info(f'Sampling a max of {n_songs} from each cluster')
		else:
			logger.info(
				'Number of songs to sample is not specified. Performing '
				'an exhaustive scan.')
		return n_songs

	def adj_rating(self, user: model.User, ne: model.User, song: str) -> int:
		"""Computes a context-based adjusted rating of a song."""
		logger.debug(
			f'Computing the adjusted rating of {song} based on user '
			f'{user.name} and their neighbor {ne.name}')

		def capacitive(r, t):
			r = np.where(r < DEFAULT_RATING, -np.exp(-t) + DEFAULT_RATING, r)
			r = np.where(r > DEFAULT_RATING, np.exp(-t) + DEFAULT_RATING, r)
			return r

		def format_(arr, label, d=3):
			u, n = round(arr[0], d), round(arr[1], d)
			return f'User (neighbor) {label}: {u} ({n})'

		def default_if_none(r, t):
			if r is None or t is None:
				value = (DEFAULT_RATING, util.NOW.timestamp())
			else:
				value = (r, t)
			return value

		result = self.db.get_ratings(user.name, ne.name, song)
		(u_rating, ne_rating), (u_time, ne_time) = result
		u_rating, u_time = default_if_none(u_rating, u_time)
		ne_rating, ne_time = default_if_none(ne_rating, ne_time)
		ratings = util.float_array([u_rating, ne_rating])
		deltas = util.float_array([util.delta(u_time), util.delta(ne_time)])
		ratings = capacitive(ratings, deltas)
		biases = util.float_array([user.bias, 1 - user.bias])
		if len(features := self.db.get_features(song, song=True)[1][0]) > 0:
			features = np.array([features])
			similarity = util.float_array([
				util.similarity(user.taste, features, self.metric),
				util.similarity(ne.taste, features, self.metric)]).flatten()
		else:
			logger.warning(
				f'Unable to find features for song {song}. Assuming 0 '
				f'similarity')
			similarity = util.float_array([0, 0])
		rating = sum(biases * ratings) * sum(biases * similarity)
		logger.debug(format_(ratings, 'rating'))
		logger.debug(format_(deltas, 'time delta'))
		logger.debug(format_(ratings, 'capacitive rating'))
		logger.debug(format_(biases, 'bias'))
		logger.debug(format_(similarity, 'similarity'))
		logger.debug(
			f'Adjusted rating of user {user.name}: {round(rating, 3)}')
		return rating

	def compute_ratings(
			self,
			user: model.User,
			ne: model.User,
			cluster: str) -> Tuple[np.ndarray, np.ndarray]:
		"""Computes the song ratings for a given user, neighbor, and cluster"""
		songs, ratings = [], []
		for song in self.db.get_songs(cluster, self.n_songs):
			songs.append(song)
			ratings.append(self.adj_rating(user, ne, song))
		logger.debug(f'Number of songs in cluster {cluster}: {len(ratings)}')
		value = (songs, ratings)
		if self.cache:
			self.db.cache(value, user=user.name, ne=ne.name, cluster=cluster)
		return np.array(songs), util.float_array(ratings)
