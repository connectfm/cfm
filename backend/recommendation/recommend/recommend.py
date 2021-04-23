import aioredis
import attr
import datetime
import json
import logging
import msgpack_numpy as mp
import numbers
import numpy as np
import random
from scipy.spatial import distance
from typing import Any, Callable, NoReturn, Tuple, Union

from backend.recommendation.recommend import exceptions

_NOW = datetime.datetime.utcnow()
# TODO(rdt17) Move these to environment variables
_DAY_IN_SECS = 86_400
_NEUTRAL_RATING = 2
_DEFAULT_NUM_CLUSTERS = 100
_NUM_RANDOM_USERS = 100

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

redis = await aioredis.create_redis_pool('redis://localhost')


def float_array(__x):
	return np.array(__x, dtype=np.float32)


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
		logger.info(f'Retrieving attributes of user {self.name}')
		attrs = [self.taste, self.bias, self.rad]
		keys = np.array([self.taste_key, self.bias_key, self.rad_key])
		if not any(values := await redis.mget(*keys)):
			raise exceptions.UserNotFoundError(f'Unable to find {self.name}')
		if not all(values):
			logger.warning(
				f'Unable to find all attributes of user {self.name}: '
				f'{[k for k, v in zip(keys, values) if v is None]}')
		for a, (k, v) in enumerate(zip(keys, values)):
			if k == self.taste_key:
				attrs[a] = float_array(mp.unpackb(v))
			else:
				attrs[a] = np.float32(v)
		if loc := redis.geopos(self.name):
			self.long, self.lat = loc
		else:
			logger.warning(f'Unable to find location of user {self.name}')

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
		return f'taste:{s}'

	@property
	def bias_key(self) -> str:
		return self.as_bias_key(self.name)

	@classmethod
	def as_bias_key(cls, s: str) -> str:
		return f'bias:{s}'

	@property
	def loc_key(self) -> str:
		return self.as_loc_key(self.name)

	@classmethod
	def as_loc_key(cls, s: str) -> str:
		return f'loc:{s}'

	@property
	def rad_key(self) -> str:
		return self.as_rad_key(self.name)

	@classmethod
	def as_rad_key(cls, s: str) -> str:
		return f'rad:{s}'

	def as_scores_key(self, user: 'User') -> str:
		return f'cscores:{self.name}.{user.name}'

	def as_ratings_key(self, user: 'User', cluster: int) -> str:
		return f'cratings:{self.name}.{user.name}.{cluster}'


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
		if (user := User.from_name(user)).taste is None:
			logger.warning(
				f'Unable to find the taste of user {user.name}. Using a taste '
				'from a random user')
			user.taste = await self.get_random_taste()
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
		if not all((user.long, user.lat, user.rad)):
			logger.warning(f'Unable to find the location of user {user.name}')
			ne = []
		else:
			ne = redis.georadius(user.name, user.long, user.lat, user.rad)
			logger.info(f'Found {len(ne)} neighbors of user {user.name}')
		return np.array(ne)

	@staticmethod
	async def get_tastes(*users: str) -> Tuple[np.ndarray, np.ndarray]:
		"""Returns the taste vectors of one or more users."""
		logger.info(f'Retrieving the music tastes of {len(users)} neighbors')
		tastes = await redis.mget(*(User.as_taste_key(u) for u in users))
		if missing := [t for t in tastes if t is None]:
			logger.warning(
				f'Unable to find tastes for {len(missing)} of {len(users)} '
				f'users. Filtering out missing tastes. The following are the '
				f'users with missing tastes: {missing}')
			present = [(u, t) for u, t in zip(users, tastes) if t is not None]
			users = np.array([u for u, _ in present])
			tastes = float_array([mp.unpackb(t) for _, t in present])
		return users, tastes

	async def get_random_taste(self) -> np.ndarray:
		logger.info(
			'Randomly retrieving the music taste from one of the first '
			f'{_NUM_RANDOM_USERS} users')
		i, stop, taste = 0, self._rng.integers(_NUM_RANDOM_USERS), None
		async for t in redis.scan(match=User.as_taste_key('*')):
			if i > stop:
				break
			else:
				taste = t
				i += 1
		if taste is None:
			raise exceptions.UserNotFoundError(
				'Unable to find the taste of any users')
		else:
			return float_array(mp.unpackb(taste))

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
		logger.info(f'Sampled neighbor {neighbor} for recommendation')
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
		key = user.as_ratings_key(neighbor, cluster)
		if cached := await self.get_cached(user, neighbor, key):
			songs, ratings = cached
			songs, ratings = np.array(songs), np.array(ratings)
		else:
			songs, ratings = await self.compute_song_ratings(
				user, neighbor, cluster)
		song = self.sample(songs, ratings)
		logger.info(f'Sampled song {song}')
		return song

	async def sample_cluster(self, user: User, neighbor: User) -> int:
		"""Returns a cluster based on user and neighbor contexts."""
		logger.info('Sampling a cluster from which to recommend a song')
		key = user.as_scores_key(neighbor)
		if scores := await self.get_cached(user, neighbor, key):
			scores = float_array(scores)
		else:
			scores = await self.compute_scores(user, neighbor)
		cluster = self.sample(np.arange(scores.size), scores)
		logger.info(f'Sampled cluster {cluster}')
		return cluster

	@staticmethod
	async def get_cached(user: User, neighbor: User, key: str) -> Any:
		"""Returns the cached value for a given user and neighbor"""
		value = None
		if data := await redis.get(key):
			data = json.loads(data)
			value, timestamp = data['value'], data['time']
			if Recommender.is_valid(timestamp):
				logger.info(
					f'Using cached {key} between user {user.name} and '
					f'neighbor {neighbor.name}')
			else:
				logger.info(
					'Clusters have been updated since caching the value '
					f'between user {user.name} and neighbor {neighbor.name}')
				value = None
		return value

	@staticmethod
	def is_valid(timestamp: float) -> bool:
		valid = True
		if c_time := await redis.get('ctime'):
			valid = timestamp > c_time
		else:
			logger.warning(
				'Unable to find the cluster scores timestamp. Assuming '
				'the associated value is still representative')
		return valid

	async def compute_scores(self, user: User, neighbor: User) -> np.ndarray:
		"""Computes cluster scores and caches the result"""
		scores = np.zeros(_DEFAULT_NUM_CLUSTERS)
		per_cluster = np.zeros(_DEFAULT_NUM_CLUSTERS)
		i = 0
		async for cluster in redis.iscan(match='cluster:*'):
			async for song in redis.sscan(f'cluster:{cluster}'):
				rating = await self.adj_rating(user, neighbor, song)
				scores[i] += rating
				per_cluster[i] += 1
			i += 1
		scores = scores[np.flatnonzero(scores)]
		per_cluster = per_cluster[np.flatnonzero(per_cluster)]
		logger.debug(f'Number of clusters: {i}')
		logger.debug(f'Number of songs: {sum(per_cluster)}')
		logger.debug(f'Number of songs per cluster: {per_cluster}')
		logger.info(f'Caching number of clusters: {i}')
		logger.info(f'Caching cluster scores: {scores}')
		key = user.as_scores_key(neighbor)
		cached = {'value': scores.tolist(), 'time': _NOW.timestamp()}
		result = await redis.set(key, cached, expire=_DAY_IN_SECS)
		self.log_cache_event(result, 'cluster scores', _DAY_IN_SECS)
		return scores

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
			f'song:{song}',
			f'rating:{user.name}.{song}',
			f'rating:{neighbor.name}.{song}',
			f'time:{user.name}.{song}',
			f'time:{neighbor.name}.{song}')
		logger.debug('Retrieved song features, ratings, and timestamps')
		ratings = float_array([u_rating, ne_rating])
		logger.debug(_format(ratings, 'rating'))
		deltas = float_array([self.delta(u_time), self.delta(ne_time)])
		logger.debug(_format(deltas, 'time delta'))
		ratings = capacitive(ratings, deltas)
		logger.debug(_format(ratings, 'capacitive rating'))
		biases = float_array([user.bias, 1 - user.bias])
		logger.debug(_format(biases, 'bias'))
		similarity = float_array([
			1 / (1 + self._metric(user.taste, features)),
			1 / (1 + self._metric(neighbor.taste, features))])
		logger.debug(_format(similarity, 'similarity'))
		rating = sum(biases * ratings) * sum(biases * similarity)
		logger.debug(
			f'Adjusted rating of user {user.name}: {round(rating, 3)}')
		return rating

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

	async def compute_song_ratings(
			self,
			user: User,
			neighbor: User,
			cluster: int) -> Tuple[np.ndarray, np.ndarray]:
		"""Computes the song ratings for a given user, neighbor, and cluster"""
		songs, ratings = [], []
		async for song in redis.sscan(f'cluster:{cluster}'):
			songs.append(song)
			ratings.append(await self.adj_rating(user, neighbor, song))
		logger.debug(f'Number of songs in cluster {cluster}: {len(ratings)}')
		logger.info(
			f'Caching the ratings for cluster {cluster} between user '
			f'{user.name} and neighbor {neighbor.name}')
		key = user.as_ratings_key(neighbor, cluster)
		cached = {'value': (songs, ratings), 'time': _NOW.timestamp()}
		result = await redis.set(key, json.dumps(cached), expire=_DAY_IN_SECS)
		self.log_cache_event(result, 'the ratings', _DAY_IN_SECS)
		return np.array(songs), np.array(ratings)
