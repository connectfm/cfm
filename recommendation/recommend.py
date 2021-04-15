import datetime
import logging
import numbers
import random
from typing import Any, Callable, Tuple

import aioredis
import attr
import numpy as np
from scipy.spatial import distance

from recommendation import exceptions

NOW = datetime.datetime.utcnow()
DAY_IN_SECS = 86_400
NEUTRAL_RATING = 2

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

redis = await aioredis.create_redis_pool('redis://localhost')


@attr.s(slots=True)
class User:
	name = attr.ib(type=str)
	taste = attr.ib(type=np.ndarray, default=None)
	bias = attr.ib(type=numbers.Real, default=None)
	lat = attr.ib(type=numbers.Real, default=None)
	long = attr.ib(type=numbers.Real, default=None)
	rad = attr.ib(type=numbers.Real, default=None)

	def __attrs_post_init__(self):
		if self.taste is not None:
			self.taste = np.array(self.taste)

	def retrieve(self):
		logger.info(f'Retrieving attributes for user {self.name}')
		attrs = [self.taste, self.bias, self.lat, self.long, self.rad]
		missing = np.array([a is None for a in attrs])
		missing = np.flatnonzero(missing)
		keys = np.array([
			f'{self.name}_taste',
			f'{self.name}_bias',
			f'{self.name}_lat',
			f'{self.name}_long',
			f'{self.name}_rad'])
		query = keys[missing]
		values = redis.mget(*query)
		if not any(values):
			raise exceptions.UserNotFoundError(f'Unable to find {self.name}')
		if not all(values):
			logger.warning(
				f'Unable to find all attributes for user {self.name}: '
				f'{[v for v in values if v is None]}')
		for m, v in zip(missing, values):
			attrs[m] = v
		if self.taste is not None:
			self.taste = np.array(self.taste)

	@classmethod
	def from_name(cls, name: str):
		user = User(name)
		user.retrieve()
		return user


@attr.s(slots=True, frozen=True)
class Recommender:
	"""Recommendation system for connect.fm"""
	metric = attr.ib(type=Callable, default=distance.euclidean)
	rng = attr.ib(type=np.random.Generator, default=np.random.default_rng())

	def __attrs_post_init__(self):
		logger.debug(f'Distance metric: {self.metric}')
		logger.debug(f'Numpy random state: {np.random.get_state()}')
		logger.debug(f'Python random state: {random.getstate()}')

	async def recommend(self, user: str) -> str:
		"""Returns a recommended song based on a user."""
		logger.info(f'Retrieving a recommendation for user {user}')
		user = User.from_name(user)
		logger.info(f'Finding the neighbors of user {user}')
		neighbors = await self.get_neighbors(user)
		if neighbors.size > 0:
			logger.info('Retrieving the music tastes of neighbors')
			neighbors, tastes = await self.get_tastes(*neighbors)
			if neighbors.size > 0:
				logger.info('Sampling a neighbor for a recommendation')
				neighbor = self.sample_neighbor(user, neighbors, tastes)
			else:
				logger.warning(
					'Unable to find any neighbors with a taste attribute. '
					f'Using user {user.name} as their own neighbor')
				# TODO(rdt17) Still possible that user does not have a taste.
				#  Maybe keep track of an average taste in redis to use as
				#  default
				neighbor = user
		else:
			logger.warning(
				f'Unable to find neighbors of user {user.name} either because '
				f'of missing attributes or because no users are present '
				f'within the set radius. Using user {user.name} as their own '
				f'neighbor')
			neighbor = user
		logger.info('Sampling a song for recommendation')
		song = await self.sample_song(user, neighbor)
		return song

	@staticmethod
	async def get_neighbors(user: User) -> np.ndarray:
		"""Returns the other nearby users of a given user."""
		geolocation = {
			'latitude': user.lat, 'longitude': user.long, 'radius': user.rad}
		if not all(geolocation):
			logger.warning(
				f'Unable to find all geolocation attributes of user'
				f' {user.name}: '
				f'{[k for k, v in geolocation.items() if v is None]}')
			ne = []
		else:
			ne = redis.georadius(user.name, *geolocation.values())
			logger.info(f'Found {len(ne)} neighbors of user {user.name}')
		return np.array(ne)

	@staticmethod
	async def get_tastes(*users: str) -> Tuple[np.ndarray, np.ndarray]:
		"""Returns the taste vectors of one or more users."""
		tastes = redis.mget(*(f'{user}_taste' for user in users))
		if not all(tastes):
			logger.warning(
				'Unable to find tastes for users: '
				f'{[t for t in tastes if t is None]}')
			logger.warning('Filtering out neighbors with missing tastes')
			present = [(u, t) for u, t in zip(users, tastes) if t is not None]
			users = [u for u, _ in present]
			tastes = [t for _, t in present]
		return np.array(users), np.array(tastes)

	def sample_neighbor(
			self,
			user: User,
			neighbors: np.ndarray,
			tastes: np.ndarray) -> User:
		"""Returns a neighbor using taste to weight the sampling."""
		u_taste = np.array([user.taste])
		dissimilarity = distance.cdist(u_taste, tastes, metric=self.metric)
		similarity = 1 / (1 + dissimilarity)
		neighbor, idx = self.sample(neighbors, similarity, with_index=True)
		logger.info(f'User {neighbor.name} sampled for a recommendation')
		neighbor = User(neighbor, taste=tastes[idx])
		return neighbor

	def sample(
			self,
			population: np.ndarray,
			weights: np.ndarray,
			with_index: bool = False) -> Any:
		"""Returns an element from the population using weighted sampling."""
		probs = weights / sum(weights)
		if with_index:
			population = np.vstack((np.arange(len(population)), population)).T
			idx_and_sample = self.rng.choice(population, p=probs)
			idx, sample = idx_and_sample[0], idx_and_sample[1:]
			sample = sample.item() if sample.shape == (1,) else sample
			sample = (sample, idx)
		else:
			sample = self.rng.choice(population, p=probs)
		return sample

	async def sample_song(self, user: User, neighbor: User) -> str:
		"""Returns a song based on user and neighbor contexts."""
		logger.info('Sampling a cluster for a song')
		cluster = await self.sample_cluster(user, neighbor)
		songs, ratings = [], []
		idx = 0
		async for song in redis.sscan(f'cluster_{cluster}'):
			rating = self.adj_rating(user, neighbor, song)
			songs[idx], ratings[idx] = song, rating
			idx += 1
		song = self.sample(np.array(songs), np.array(ratings))
		return song

	async def sample_cluster(self, user: User, neighbor: User) -> int:
		"""Returns a cluster based on user and neighbor contexts."""
		n_clusters = await redis.get('n_clusters')
		if n_clusters is None:
			logger.warning('Unable to find the number of clusters. Using 50')
			n_clusters = 50
		# TODO(rdt17) Left off here
		c_ratings = np.zeros(n_clusters)
		idx = 0
		async for cluster in redis.iscan('cluster_*'):
			async for song in redis.sscan(f'cluster_{cluster}'):
				rating = self.adj_rating(user, neighbor, song)
				c_ratings[idx] += rating
			idx += 1
		cluster = self.sample(np.arange(n_clusters), c_ratings)
		return cluster

	def adj_rating(self, user: User, neighbor: User, song: str) -> int:
		"""Computes a context-based adjusted rating of a song."""

		def capacitive(r, t):
			r = np.where(r < NEUTRAL_RATING, -np.exp(-t) + NEUTRAL_RATING, r)
			r = np.where(r > NEUTRAL_RATING, np.exp(-t) + NEUTRAL_RATING, r)
			return r

		features, u_rating, ne_rating, u_time, ne_time = await redis.mget(
			song,
			f'{user.name}_{song}_rating',
			f'{neighbor.name}_{song}_rating',
			f'{user.name}_{song}_time',
			f'{neighbor.name}_{song}_time')
		ratings = np.array([u_rating, ne_rating])
		deltas = np.array([self.delta(u_time), self.delta(ne_time)])
		ratings = capacitive(ratings, deltas)
		biases = np.array([user.bias, 1 - user.bias])
		dists = np.array([
			self.metric(user.taste, features),
			self.metric(neighbor.taste, features)])
		num = sum(biases * ratings)
		den = sum(biases * dists)
		rating = num / den
		return rating

	@staticmethod
	def delta(timestamp: float) -> float:
		"""Returns the difference in days between a timestamp and now."""
		delta = NOW - datetime.datetime.utcfromtimestamp(timestamp)
		delta = delta.total_seconds() / DAY_IN_SECS
		return delta
