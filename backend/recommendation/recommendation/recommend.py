import logging
import random
from typing import Any, Callable, Tuple

import attr
import numpy as np
from scipy.spatial import distance

import model
import util

DEFAULT_RATING = 2

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


@attr.s(slots=True)
class Recommender:
	"""Recommendation system for connect.fm"""
	db = attr.ib(type=model.RecommendDB)
	metric = attr.ib(type=Callable, default=distance.euclidean)
	max_clusters = attr.ib(type=int, default=100)
	seed = attr.ib(type=Any, default=None)
	_rng = attr.ib(type=np.random.Generator, init=False, repr=False)

	def __attrs_post_init__(self):
		random.seed(self.seed)
		self._rng = np.random.default_rng(self.seed)
		logger.debug(f'Distance metric: {self.metric}')
		logger.debug(f'Seed: {self.seed}')

	def recommend(self, name: str) -> str:
		"""Returns a recommended song based on a user."""
		logger.info(f'Retrieving a recommendation for user {name}')
		if (user := self.db.get_user(name)).taste is None:
			logger.warning(
				f'Unable to find the taste of user {user.name}. Using a taste '
				'from a random user')
			user.taste = self.db.get_random_taste()
		if (neighbors := self.db.get_neighbors(user)).size > 0:
			neighbors, tastes = self.db.get_features(*neighbors, songs=False)
			if neighbors.size > 0:
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
		logger.info(f'Sampling 1 of {neighbors.size} neighbors of {user.name}')
		u_taste = np.array([user.taste])
		dissimilarity = distance.cdist(u_taste, tastes, metric=self.metric)
		similarity = 1 / (1 + dissimilarity)
		ne, idx = util.sample(neighbors, similarity, with_index=True)
		logger.info(f'Sampled neighbor {ne}')
		return model.User(ne, taste=tastes[idx])

	def sample_song(self, user: model.User, ne: model.User) -> str:
		"""Returns a song based on user and neighbor contexts."""
		cluster = self.sample_cluster(user, ne)
		logger.info(f'Sampling a song to recommend')
		key = self.db.to_ratings_key(user.name, ne.name, cluster)
		if cached := self.db.get_cached(key):
			songs, ratings = cached
			songs, ratings = np.array(songs), util.float_array(ratings)
		else:
			songs, ratings = self.compute_ratings(user, ne, cluster)
		song = util.sample(songs, ratings)
		logger.info(f'Sampled song {song}')
		return song

	def sample_cluster(self, user: model.User, ne: model.User) -> int:
		"""Returns a cluster based on user and neighbor contexts."""
		logger.info('Sampling a cluster from which to recommend a song')
		key = self.db.to_scores_key(user.name, ne.name)
		if cached := self.db.get_cached(key):
			clusters, scores = cached
			clusters, scores = np.array(clusters), util.float_array(scores)
		else:
			clusters, scores = self.compute_scores(user, ne)
		cluster = util.sample(clusters, scores)
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
		clusters, scores, per_cluster = [], [], []
		for i, cluster in enumerate(self.db.get_clusters()):
			clusters.append(cluster)
			scores.append(0)
			per_cluster.append(0)
			for song in self.db.get_songs(cluster):
				scores[i] += self.adj_rating(user, ne, song)
				per_cluster[i] += 1
		logger.debug(f'Number of clusters: {len(clusters)}')
		logger.debug(f'Number of songs: {sum(per_cluster)}')
		logger.debug(f'Number of songs per cluster: {per_cluster}')
		key = self.db.to_scores_key(user.name, ne.name)
		self.db.cache(key, (clusters, scores))
		return np.array(clusters), util.float_array(scores)

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
		if (features := self.db.get_features(song, songs=True)) is not None:
			similarity = util.float_array([
				1 / (1 + self.metric(user.taste, features)),
				1 / (1 + self.metric(ne.taste, features))])
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
			cluster: int) -> Tuple[np.ndarray, np.ndarray]:
		"""Computes the song ratings for a given user, neighbor, and cluster"""
		songs, ratings = [], []
		for song in self.db.get_songs(cluster):
			songs.append(song)
			ratings.append(self.adj_rating(user, ne, song))
		logger.debug(f'Number of songs in cluster {cluster}: {len(ratings)}')
		key = self.db.to_ratings_key(user.name, ne.name, cluster)
		self.db.cache(key, (songs, ratings))
		return np.array(songs), util.float_array(ratings)
