import attr
import logging
import numpy as np
import random
from scipy.spatial import distance
from typing import Any, Callable, Tuple

import model
import util

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
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
			neighbors, tastes = self.db.get_tastes(*neighbors)
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
		logger.info(f'Sampling a song to recommendation')
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
		logger.info('Sampling a cluster from which to recommendation a song')
		key = self.db.to_scores_key(user.name, ne.name)
		if scores := self.db.get_cached(key):
			scores = util.float_array(scores)
		else:
			scores = self.compute_scores(user, ne)
		cluster = util.sample(np.arange(scores.size), scores)
		logger.info(f'Sampled cluster {cluster}')
		return cluster

	def compute_scores(self, user: model.User, ne: model.User) -> np.ndarray:
		"""Computes cluster scores and caches the result"""
		logger.info(
			f'Computing cluster scores between user {user.name} and neighbor '
			f'{ne.name}')
		scores = np.zeros(self.max_clusters)
		per_cluster = np.zeros(self.max_clusters)
		i = 0
		for cluster in self.db.get_clusters():
			for song in self.db.get_songs(cluster):
				scores[i] += self.adj_rating(user, ne, song)
				per_cluster[i] += 1
			i += 1
		scores = scores[np.flatnonzero(scores)]
		per_cluster = per_cluster[np.flatnonzero(per_cluster)]
		logger.debug(f'Number of clusters: {i}')
		logger.debug(f'Number of songs: {sum(per_cluster)}')
		logger.debug(f'Number of songs per cluster: {per_cluster}')
		key = self.db.to_scores_key(user.name, ne.name)
		self.db.cache(key, scores.tolist())
		return scores

	def adj_rating(self, user: model.User, ne: model.User, song: str) -> int:
		"""Computes a context-based adjusted rating of a song."""
		logger.debug(
			f'Computing the adjusted rating of {song} based on user '
			f'{user.name} and their neighbor {ne.name}')

		def capacitive(r, t):
			r = np.where(r < 2, -np.exp(-t) + 2, r)
			r = np.where(r > 2, np.exp(-t) + 2, r)
			return r

		def _format(arr, label, d=3):
			u, n = round(arr[0], d), round(arr[1], d)
			return f'User (neighbor) {label}: {u} ({n})'

		result = self.db.get_features_and_ratings(user.name, ne.name, song)
		features, (u_rating, ne_rating), (u_time, ne_time) = result
		ratings = util.float_array([u_rating, ne_rating])
		logger.debug(_format(ratings, 'rating'))
		deltas = util.float_array([util.delta(u_time), util.delta(ne_time)])
		logger.debug(_format(deltas, 'time delta'))
		ratings = capacitive(ratings, deltas)
		logger.debug(_format(ratings, 'capacitive rating'))
		biases = util.float_array([user.bias, 1 - user.bias])
		logger.debug(_format(biases, 'bias'))
		similarity = util.float_array([
			1 / (1 + self.metric(user.taste, features)),
			1 / (1 + self.metric(ne.taste, features))])
		logger.debug(_format(similarity, 'similarity'))
		rating = sum(biases * ratings) * sum(biases * similarity)
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
