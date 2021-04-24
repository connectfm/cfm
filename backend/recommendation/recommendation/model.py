import aioredis
import attr
import json
import logging
import msgpack_numpy as mp
import numbers
import numpy as np
import random
from typing import Any, AsyncIterable, Sequence, Tuple, Union

import util

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


@attr.s(slots=True)
class User:
	name = attr.ib(type=str)
	taste = attr.ib(type=np.ndarray, default=None)
	bias = attr.ib(type=numbers.Real, default=None)
	lat = attr.ib(type=numbers.Real, default=None)
	long = attr.ib(type=numbers.Real, default=None)
	rad = attr.ib(type=numbers.Real, default=None)


@attr.s(slots=True)
class RecommendDB:
	seed = attr.ib(type=Any, default=None)
	_rng = attr.ib(type=np.random.Generator, init=False, repr=False)
	_redis = attr.ib(type=aioredis.Redis, init=False, repr=False)

	def __attrs_post_init__(self):
		random.seed(self.seed)
		self._rng = np.random.default_rng(self.seed)
		logger.debug(f'Seed: {self.seed}')

	async def __aenter__(self):
		self._redis = await aioredis.create_redis_pool(('localhost', 6379))
		return self

	async def __aexit__(self, exc_type, exc_val, exc_tb):
		self._redis.close()
		await self._redis.wait_closed()

	async def get(self, *k: str) -> Any:
		if isinstance(k, Sequence):
			value = self._redis.mget(*k, encoding='utf-8')
		else:
			value = self._redis.get(k, encoding='utf-8')
		return value

	async def set(self, k: str, v: str, exp: int = util.DAY_IN_SECS) -> bool:
		return self._redis.set(k, v, expire=exp)

	async def cache(self, key: str, value: Any, exp: int = util.DAY_IN_SECS):
		logger.debug(f'Caching {key} with value {value}')
		serialized = json.dumps({'value': value, 'time': util.NOW.timestamp()})
		result = await self.set(key, serialized, exp=exp)
		self._log_cache_event(result, key, exp)

	@staticmethod
	def _log_cache_event(result: bool, name: str, exp: int = None):
		if result:
			if exp is None:
				logger.info(f'Successfully cached {name} without expiration')
			else:
				logger.info(f'Successfully cached {name} for {exp} seconds')
		else:
			logger.warning(f'Failed to cache {name}')

	async def get_cached(self, k: str) -> Any:
		"""Returns the cached value"""
		value = None
		if data := await self.get(k):
			data = json.loads(data)
			value, timestamp = data['value'], data['time']
			if await self._is_valid(timestamp):
				logger.info(f'Using cached the value of {k}')
			else:
				logger.warning(f'Clusters have been updated since caching {k}')
		else:
			logger.warning(f'Cached value of {k} does not exist')
		return value

	async def _is_valid(self, timestamp: float) -> bool:
		valid = True
		if c_time := await self.get_clusters_expiration():
			valid = timestamp > c_time
		else:
			logger.warning(
				'Unable to find the cluster scores timestamp. Assuming '
				'the associated value is still representative')
		return valid

	def get_songs(self, cluster: Union[str, int]) -> AsyncIterable:
		return self._redis.sscan(f'cluster:{cluster}')

	def get_clusters(self) -> AsyncIterable:
		return self._redis.iscan(match='cluster:*')

	async def get_clusters_expiration(self):
		return await self.get('ctime')

	async def get_feats_and_ratings(
			self, user: str, ne: str, song: str) -> Tuple:
		logger.debug('Retrieving song features, ratings, and timestamps')
		feats, u_rating, ne_rating, u_time, ne_time = await self.get(
			self.to_song_key(song),
			self.to_rating_key(user, song),
			self.to_rating_key(ne, song),
			self.to_time_key(user, song),
			self.to_time_key(ne, song))
		logger.debug('Retrieved song features, ratings, and timestamps')
		return feats, (u_rating, ne_rating), (u_time, ne_time)

	async def get_random_taste(self, n: int = 100):
		logger.info(
			'Randomly retrieving the music taste from one of the first '
			f'{n} users')
		i, stop, taste = 0, self._rng.integers(n), None
		async for t in self._redis.scan(match=self.to_taste_key('*')):
			if i > stop:
				break
			else:
				taste = t
				i += 1
		if taste is None:
			raise KeyError('Unable to find the taste of any users')
		else:
			return util.float_array(mp.unpackb(taste))

	async def get_tastes(self, *users: str) -> Tuple[np.ndarray, np.ndarray]:
		"""Returns the taste vectors of one or more users."""
		logger.info(f'Retrieving the music tastes of {len(users)} neighbors')
		tastes = await self.get(*(self.to_taste_key(u) for u in users))
		if missing := [t for t in tastes if t is None]:
			logger.warning(
				f'Unable to find tastes for {len(missing)} of {len(users)} '
				f'users. Filtering out missing tastes. The following are the '
				f'users with missing tastes: {missing}')
			present = [(u, t) for u, t in zip(users, tastes) if t is not None]
			users = np.array([u for u, _ in present])
			tastes = util.float_array([mp.unpackb(t) for _, t in present])
		return users, tastes

	async def get_neighbors(self, user: User) -> np.ndarray:
		"""Returns the other nearby users of a given user."""
		logger.info(f'Finding the neighbors of user {user.name}')
		if all((user.long, user.lat, user.rad)):
			ne = self._redis.georadius(
				user.name, user.long, user.lat, user.rad)
			logger.info(f'Found {len(ne)} neighbors of user {user.name}')
		else:
			logger.warning(f'Unable to find the location of user {user.name}')
			ne = []
		return np.array(ne)

	async def get_user(self, name: str) -> User:
		"""Returns a user with all associated attributes populated"""
		logger.info(f'Retrieving attributes of user {name}')
		keys = (
			self.to_taste_key(name),
			self.to_bias_key(name),
			self.to_radius_key(name))
		if not any(values := await self.get(*keys)):
			raise KeyError(f'Unable to find user {name}')
		if not all(values):
			logger.warning(
				f'Unable to find all attributes of user {name}: '
				f'{[k for k, v in zip(keys, values) if v is None]}')
		taste, bias, radius = values
		key = self.get_geolocation_key()
		if not (loc := await self._redis.geopos(key, name)):
			logger.warning(f'Unable to find the location of user {name}')
		return User(
			name=name,
			taste=util.float_array(mp.unpackb(taste)),
			bias=np.float32(bias),
			long=np.float64(loc[0]),
			lat=np.float64(loc[1]),
			rad=np.float32(radius))

	@classmethod
	def get_geolocation_key(cls):
		return 'location'

	@classmethod
	def to_radius_key(cls, u: str) -> str:
		return f'rad:{u}'

	@classmethod
	def to_scores_key(cls, user: str, ne: str) -> str:
		return f'cscores:{user}.{ne}'

	@classmethod
	def to_ratings_key(cls, user: str, ne: str, cluster: int) -> str:
		return f'cratings:{user}.{ne}.{cluster}'

	@classmethod
	def to_bias_key(cls, user: str) -> str:
		return f'bias:{user}'

	@classmethod
	def to_taste_key(cls, user: str) -> str:
		return f'taste:{user}'

	@classmethod
	def to_song_key(cls, song: str) -> str:
		return f'song:{song}'

	@classmethod
	def to_rating_key(cls, user: str, song: str) -> str:
		return f'rating:{user}.{song}'

	@classmethod
	def to_time_key(cls, user: str, song: str) -> str:
		return f'time:{user}.{song}'
