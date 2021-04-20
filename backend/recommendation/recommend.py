import aioredis
import attr
import datetime
import logging
import msgpack_numpy as mp
import numbers
import numpy as np
import random
from scipy.spatial import distance
from typing import Any, Callable, NoReturn, Tuple, Union

from backend.recommendation import exceptions

_NOW = datetime.datetime.utcnow()
_DAY_IN_SECS = 86_400
_NEUTRAL_RATING = 2
_DEFAULT_NUM_CLUSTERS = 50

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

	def retrieve(self) -> NoReturn:
		logger.info(f'Retrieving attributes for user {self.name}')
		attrs = [self.taste, self.bias, self.lat, self.long, self.rad]
		missing = np.array([a is None for a in attrs])
		missing = np.flatnonzero(missing)
		keys = np.array([
			self.taste_key,
			self.bias_key,
			self.lat_key,
			self.long_key,
			self.rad_key])
		query = keys[missing]
		if not any(values := redis.mget(*query)):
			raise exceptions.UserNotFoundError(f'Unable to find {self.name}')
		if not all(values):
			logger.warning(
				f'Unable to find all attributes for user {self.name}: '
				f'{[v for v in values if v is None]}')
		for m, v in zip(missing, values):
			attrs[m] = np.float32(v)
		if self.taste is None:
			self.taste = np.array(await redis.get('avg_taste'))
			logger.warning(
				f'Unable to find taste of user {self.name}. Using average '
				f'taste {self.taste}')
		else:
			self.taste = np.array(self.taste)

	@classmethod
	def from_name(cls, name: str) -> 'User':
		user = User(name)
		user.retrieve()
		return user

	@property
	def taste_key(self) -> str:
		return self.as_taste_key(self.name)

	@classmethod
	def as_taste_key(cls, s: str) -> str:
		return f'{s}_taste'

	@property
	def bias_key(self) -> str:
		return self.as_bias_key(self.name)

	@classmethod
	def as_bias_key(cls, s: str) -> str:
		return f'{s}_bias'

	@property
	def lat_key(self) -> str:
		return self.as_lat_key(self.name)

	@classmethod
	def as_lat_key(cls, s: str) -> str:
		return f'{s}_lat'

	@property
	def long_key(self) -> str:
		return self.as_long_key(self.name)

	@classmethod
	def as_long_key(cls, s: str) -> str:
		return f'{s}_long'

	@property
	def rad_key(self) -> str:
		return self.as_rad_key(self.name)

	@classmethod
	def as_rad_key(cls, s: str) -> str:
		return f'{s}_rad'

	def as_clusters_key(self, user: 'User') -> str:
		return f'{self.name}_{user.name}'

	def as_cluster_key(self, user: 'User', cluster: Union[str, int]) -> str:
		return f'{self.name}_{user.name}_{cluster}'


@attr.s(slots=True, frozen=True)
class Recommender:
	"""Recommendation system for connect.fm"""
	_metric = attr.ib(type=Callable, default=distance.euclidean)
	_rng = attr.ib(type=np.random.Generator, default=np.random.default_rng())

	def __attrs_post_init__(self):
		logger.debug(f'Distance metric: {self._metric}')
		logger.debug(f'Numpy random state: {np.random.get_state()}')
		logger.debug(f'Python random state: {random.getstate()}')

	async def recommend(self, user: str) -> str:
		"""Returns a recommended song based on a user."""
		logger.info(f'Retrieving a recommendation for user {user}')
		user = User.from_name(user)
		if (neighbors := await self.get_neighbors(user)).size > 0:
			neighbors, tastes = await self.get_tastes(*neighbors)
			if neighbors.size > 0:
				neighbor = self.sample_neighbor(user, neighbors, tastes)
			else:
				logger.warning(
					'Unable to find any neighbors with a taste attribute. '
					f'Using user {user.name} as their own neighbor')
				neighbor = user
		else:
			logger.warning(
				f'Unable to find neighbors of user {user.name} either because '
				f'of missing attributes or because no users are present '
				f'within the set radius. Using user {user.name} as their own '
				f'neighbor')
			neighbor = user
		song = await self.sample_song(user, neighbor)
		return song

	@staticmethod
	async def get_neighbors(user: User) -> np.ndarray:
		"""Returns the other nearby users of a given user."""
		logger.info(f'Finding the neighbors of user {user.name}')
		geolocation = (user.long, user.lat, user.rad)
		if not all(geolocation):
			keys = (user.long_key, user.lat_key, user.rad_key)
			logger.warning(
				f'Unable to find all geolocation attributes of user'
				f' {user.name}: '
				f'{[k for k, g in zip(keys, geolocation) if g is None]}')
			ne = []
		else:
			ne = redis.georadius(user.name, *geolocation)
			logger.info(f'Found {len(ne)} neighbors of user {user.name}')
		return np.array(ne)

	async def get_tastes(self, *users: str) -> Tuple[np.ndarray, np.ndarray]:
		"""Returns the taste vectors of one or more users."""
		logger.info(f'Retrieving the music tastes of {len(users)} neighbors')
		tastes = await redis.mget(*(User.as_taste_key(u) for u in users))
		if missing := [t for t in tastes if t is None]:
			logger.warning(
				f'Unable to find tastes for {len(missing)} users: {missing}')
			logger.warning('Filtering out neighbors with missing tastes')
			present = [(u, t) for u, t in zip(users, tastes) if t is not None]
			users = [u for u, _ in present]
			tastes = [t for _, t in present]
		return np.array(users), self.float_array(tastes)

	def sample_neighbor(
			self,
			user: User,
			neighbors: np.ndarray,
			tastes: np.ndarray) -> User:
		"""Returns a neighbor using taste to weight the sampling."""
		logger.info(
			f'Sampling 1 of {neighbors.size} neighbors of {user.name} for a '
			f'recommendation')
		u_taste = np.array([user.taste])
		dissimilarity = distance.cdist(u_taste, tastes, metric=self._metric)
		similarity = 1 / (1 + dissimilarity)
		neighbor, idx = self.sample(neighbors, similarity, with_index=True)
		logger.info(f'Sampled neighbor {neighbor.name} for recommendation')
		neighbor = User(neighbor, taste=tastes[idx])
		return neighbor

	def sample(
			self,
			population: np.ndarray,
			weights: np.ndarray,
			with_index: bool = False) -> Any:
		"""Returns an element from the population using weighted sampling."""
		probs = weights / sum(weights)
		mean = np.round(np.average(probs), 3)
		sd = np.round(np.std(probs), 3)
		logger.debug(f'Mean (sd) probability: {mean} ({sd})')
		if with_index:
			population = np.vstack((np.arange(population.size), population)).T
			idx_and_sample = self._rng.choice(population, p=probs)
			idx, sample = idx_and_sample[0], idx_and_sample[1:]
			sample = sample.item() if sample.shape == (1,) else sample
			logger.debug(f'Sampled element (index): {sample} ({idx})')
			sample = (sample, idx)
		else:
			sample = self._rng.choice(population, p=probs)
			logger.debug(f'Sampled element: {sample}')
		return sample

	async def sample_song(self, user: User, neighbor: User) -> str:
		"""Returns a song based on user and neighbor contexts."""
		cluster = await self.sample_cluster(user, neighbor)
		logger.info(f'Sampling a song to recommend')
		songs, ratings = self.get_cached_ratings(user, neighbor, cluster)
		if not all((songs, ratings)):
			songs, ratings = self.compute_song_ratings(user, neighbor, cluster)
		song = self.sample(songs, ratings)
		logger.info(f'Sampled song {song}')
		return song

	async def sample_cluster(self, user: User, neighbor: User) -> int:
		"""Returns a cluster based on user and neighbor contexts."""
		logger.info('Sampling a cluster from which to recommend a song')
		n_clusters, c_rates = self.get_all_cached_ratings(user, neighbor)
		if not c_rates:
			c_rates = self.compute_cluster_ratings(user, neighbor, n_clusters)
		cluster = self.sample(np.arange(n_clusters), c_rates)
		logger.info(f'Sampled cluster {cluster}')
		return cluster

	@staticmethod
	async def get_all_cached_ratings(user: User, neighbor: User) -> Tuple:
		if n_clusters := await redis.get('n_clusters'):
			logger.info(f'Using cached number of clusters: {n_clusters}')
			n_clusters = int(n_clusters)
		else:
			logger.warning(
				f'Unable to find the number of clusters. Using '
				f'{(n_clusters := _DEFAULT_NUM_CLUSTERS)}')
		if c_ratings := await redis.get(user.as_clusters_key(neighbor)):
			logger.info(
				f'Using cached cluster ratings between user {user.name} and '
				f'neighbor {neighbor.name}')
			c_ratings: np.ndarray = mp.unpackb(c_ratings)
			if (n_ratings := c_ratings.size) != n_clusters:
				logger.warning(
					f'Cached number of clusters ({n_clusters}) and number of '
					f'cluster ratings ({n_ratings}) do not match. Using '
					f'{(n_clusters := n_ratings)} number of clusters')
		return n_clusters, c_ratings

	async def compute_cluster_ratings(
			self, user: User, neighbor: User, n_clusters: int) -> np.ndarray:
		c_ratings, per_cluster = np.zeros(n_clusters), np.zeros(n_clusters)
		idx = 0
		async for cluster in redis.iscan(match='cluster_*'):
			async for song in redis.sscan(f'cluster_{cluster}'):
				rating = await self.adj_rating(user, neighbor, song)
				c_ratings[idx] += rating
				per_cluster[idx] += 1
			idx += 1
		c_ratings = c_ratings[np.flatnonzero(c_ratings)]
		per_cluster = per_cluster[np.flatnonzero(per_cluster)]
		logger.debug(f'Number of clusters: {idx}')
		logger.debug(f'Number of songs in clusters: {sum(per_cluster)}')
		logger.debug(f'Number of songs per cluster: {per_cluster}')
		logger.info(f'Caching number of clusters: {idx}')
		result = await redis.set('n_clusters', idx)
		self.log_cache_event(result, 'the number of clusters')
		logger.info(f'Caching cluster ratings: {c_ratings}')
		key = user.as_clusters_key(neighbor)
		result = await redis.set(key, mp.packb(c_ratings), expire=_DAY_IN_SECS)
		self.log_cache_event(result, 'cluster ratings', _DAY_IN_SECS)
		return c_ratings

	async def adj_rating(self, user: User, neighbor: User, song: str) -> int:
		"""Computes a context-based adjusted rating of a song."""
		logger.debug(
			f'Computing the adjusted rating of {song} based on user '
			f'{user.name} and their neighbor {neighbor.name}')

		def capacitive(r, t):
			r = np.where(r < _NEUTRAL_RATING, -np.exp(-t) + _NEUTRAL_RATING, r)
			r = np.where(r > _NEUTRAL_RATING, np.exp(-t) + _NEUTRAL_RATING, r)
			return r

		def _format(x, label, d=3):
			u, n = round(x[0], d), round(x[1], d)
			return f'User (neighbor) {label}: {u} ({n})'

		logger.debug('Retrieving song features, ratings, and timestamps')
		features, u_rating, ne_rating, u_time, ne_time = await redis.mget(
			f'song_{song}',
			f'{user.name}_{song}_rating',
			f'{neighbor.name}_{song}_rating',
			f'{user.name}_{song}_time',
			f'{neighbor.name}_{song}_time')
		logger.debug('Retrieved song features, ratings, and timestamps')
		ratings = self.float_array([u_rating, ne_rating])
		logger.debug(_format(ratings, 'rating'))
		deltas = self.float_array([self.delta(u_time), self.delta(ne_time)])
		logger.debug(_format(deltas, 'time delta'))
		ratings = capacitive(ratings, deltas)
		logger.debug(_format(ratings, 'capacitive rating'))
		biases = self.float_array([user.bias, 1 - user.bias])
		logger.debug(_format(biases, 'bias'))
		similarity = self.float_array([
			1 / (1 + self._metric(user.taste, features)),
			1 / (1 + self._metric(neighbor.taste, features))])
		logger.debug(_format(similarity, 'similarity'))
		rating = sum(biases * ratings) * sum(biases * similarity)
		logger.debug(
			f'Adjusted rating of user {user.name}: {round(rating, 3)}')
		return rating

	@staticmethod
	def float_array(__x):
		return np.array(__x, dtype=np.float32)

	@staticmethod
	def delta(timestamp: Union[str, float]) -> float:
		"""Returns the difference in days between a timestamp and now."""
		timestamp = float(timestamp)
		logger.debug(f'Computing time delta of timestamp {timestamp}')
		delta = _NOW - datetime.datetime.utcfromtimestamp(timestamp)
		delta = delta.total_seconds() / _DAY_IN_SECS
		logger.debug(f'Computed time delta (days): {delta}')
		return delta

	@staticmethod
	def log_cache_event(result: bool, name: str, expire: int = None):
		if result:
			if expire is None:
				logger.info(f'Successfully cached {name} without expiration')
			else:
				logger.info(f'Successfully cached {name} for {expire} seconds')
		else:
			logger.warning(f'Failed to cache {name}')

	@staticmethod
	async def get_cached_ratings(
			user: User, neighbor: User, cluster: int) -> Tuple:
		if ratings := await redis.get(user.as_cluster_key(neighbor, cluster)):
			logger.info(
				f'Using cached song ratings for cluster {cluster} between '
				f'user {user.name} and neighbor {neighbor.name}')
			ratings: np.ndarray = mp.unpackb(ratings)
			songs = np.array(list(await redis.smembers(f'cluster_{cluster}')))
			if (n_ratings := ratings.size) != (n_songs := songs.size):
				trunc = min(n_ratings, n_songs)
				songs, ratings = songs[trunc], ratings[trunc]
				logger.warning(
					f'Number of ratings ({n_ratings}) does not equal the '
					f'number of songs ({n_songs}). Using the first {trunc} '
					f'songs and ratings')
		else:
			songs, ratings = None, None
		return songs, ratings

	async def compute_song_ratings(
			self, user: User, neighbor: User, cluster: int) -> Tuple:
		songs, ratings = [], []
		idx = 0
		async for song in redis.sscan(f'cluster_{cluster}'):
			rating = self.adj_rating(user, neighbor, song)
			songs[idx], ratings[idx] = song, rating
			idx += 1
		logger.debug(f'Number of songs in cluster {cluster}: {idx}')
		songs, ratings = np.array(songs), np.array(ratings)
		logger.info(
			f'Caching the ratings for cluster {cluster} between user '
			f'{user.name} and neighbor {neighbor.name}')
		key = user.as_cluster_key(neighbor, cluster)
		ratings = mp.packb(ratings)
		result = await redis.set(key, ratings, expire=_DAY_IN_SECS)
		self.log_cache_event(result, 'the ratings', _DAY_IN_SECS)
		return songs, ratings
