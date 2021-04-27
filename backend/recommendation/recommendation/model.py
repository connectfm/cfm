import codecs
from numbers import Real
import pickle
import random
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import attr
import msgpack_numpy as mp
import numpy as np
import redis

import util

logger = util.get_logger(__name__)


@attr.s(slots=True)
class User:
	name = attr.ib(type=str)
	taste = attr.ib(type=np.ndarray, default=None)
	bias = attr.ib(type=Real, default=None)
	lat = attr.ib(type=Real, default=None)
	long = attr.ib(type=Real, default=None)
	rad = attr.ib(type=Real, default=None)


@attr.s(slots=True)
class RecommendDB:
	seed = attr.ib(type=Any, default=None)
	_rng = attr.ib(type=np.random.Generator, init=False, repr=False)
	_redis = attr.ib(type=redis.Redis, init=False, repr=False)

	def __attrs_post_init__(self):
		random.seed(self.seed)
		self._rng = np.random.default_rng(self.seed)
		logger.debug(f'Seed: {self.seed}')

	def __enter__(self):
		self._redis = redis.Redis()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self._redis.close()

	def get(self, *keys: str, decoder: Union[Callable, str] = None) -> Tuple:
		return tuple(
			v if v is None else self._decode(v, decoder)
			for v in self._redis.mget(*keys))

	@staticmethod
	def _decode(value, decoder: Union[Callable, str]):
		if isinstance(decoder, str):
			value = codecs.decode(value, decoder)
		elif isinstance(decoder, Callable):
			value = decoder(value)
		return value

	def set(
			self,
			keys: Union[str, Iterable[str]],
			values: Any,
			*,
			expire: int = None,
			encoder: Union[Callable, str] = None) -> bool:
		if isinstance(keys, str):
			encoded = self._encode(values, encoder)
			result = self._redis.set(keys, encoded, ex=expire)
		else:
			items = {k: self._encode(v, encoder) for k, v in zip(keys, values)}
			result = self._redis.mset(items)
			if expire is not None:
				for k in items:
					self._redis.expire(k, expire)
		return result

	@staticmethod
	def _encode(value: Any, encoder: Union[Callable, str]):
		if isinstance(encoder, str):
			value = codecs.encode(value, encoder)
		elif isinstance(encoder, Callable):
			value = encoder(value)
		return value

	def get_bias(self, *names: str) -> Sequence[Optional[Real]]:
		return self._redis.mget(*(self.to_bias_key(n) for n in names))

	def set_bias(
			self,
			names: Union[str, Iterable[str]],
			values: Union[Real, Iterable[Real]],
			expire: int = None) -> bool:
		keys = (self.to_bias_key(n) for n in names)
		return self.set(keys, values, expire=expire)

	def get_radius(self, *names: str) -> Sequence[Optional[Real]]:
		return self._redis.mget(*(self.to_radius_key(n) for n in names))

	def set_radius(
			self,
			names: Union[str, Iterable[str]],
			values: Union[Real, Iterable[Real]],
			expire: int = None) -> bool:
		keys = (self.to_radius_key(n) for n in names)
		return self.set(keys, values, expire=expire)

	def get_location(self, *names: str) -> Sequence[Tuple[Real, Real]]:
		return self._redis.geopos(self.get_location_key(), *names)

	def set_location(
			self,
			names: Union[str, Iterable[str]],
			longs: Union[Real, Iterable[Real]],
			lats: Union[Real, Iterable[Real]]) -> Sequence[bool]:
		key = self.get_location_key()
		result = []
		for name, long, lat in zip(names, longs, lats):
			result.append(self._redis.geoadd(key, *(long, lat, name)))
		return tuple(result)

	def get_clusters(self) -> Iterable[str]:
		keys = self._redis.scan_iter(match=self.to_cluster_key('*'))
		return (self._get_name(k) for k in keys)

	@staticmethod
	def _get_name(encoded: bytes):
		return encoded.decode('utf-8').split(':')[-1]

	def set_cluster(self, name: str, *songs: Any) -> bool:
		songs = (self.to_song_key(s) for s in songs)
		return self._redis.sadd(self.to_cluster_key(name), *songs)

	def get_clusters_time(self) -> float:
		key = self.get_clusters_time_key()
		return self.get(key, decoder=util.float_decoder)[0]

	def set_clusters_time(self, timestamp: float) -> bool:
		return self._redis.set(self.get_clusters_time_key(), timestamp)

	def cache(
			self,
			key: str,
			value: Any,
			*,
			expire: int = None,
			encoder: Union[Callable, str] = None) -> bool:
		logger.debug(f'Caching {key} with value {value}')
		value = self._encode(value, encoder)
		value = {'value': value, 'time': util.NOW.timestamp()}
		result = self.set(key, value, encoder=pickle.dumps, expire=expire)
		self._log_cache_event(result, key, expire)
		return result

	@staticmethod
	def _log_cache_event(result: bool, name: str, expire: int = None):
		if result:
			if expire is None:
				logger.info(f'Successfully cached {name} without expiration')
			else:
				logger.info(f'Successfully cached {name} for {expire} seconds')
		else:
			logger.warning(f'Failed to cache {name}')

	def get_cached(self, key: str) -> Any:
		if data := self.get(key, decoder=pickle.loads)[0]:
			value, timestamp = data['value'], data['time']
			if self._is_valid(timestamp):
				logger.info(f'Using cached the value of {key}')
			else:
				logger.warning(
					f'Clusters have been updated since caching {key}')
		else:
			logger.warning(f'Cached value of {key} does not exist')
			value = None
		return value

	def _is_valid(self, timestamp: float) -> bool:
		if c_time := self.get_clusters_time():
			valid = timestamp > c_time
		else:
			logger.warning(
				'Unable to find the cluster scores timestamp. Assuming '
				'the associated value is still representative')
			valid = True
		return valid

	def get_songs(self, cluster: str) -> Iterable[str]:
		key = self.to_cluster_key(cluster)
		return (self._get_name(s) for s in self._redis.sscan_iter(key))

	def get_ratings(
			self,
			user: str,
			ne: str,
			song: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
		logger.debug(
			f'Retrieving ratings, and timestamps for user {user} and neighbor '
			f'{ne}')
		u_rating, ne_rating, u_time, ne_time = self.get(
			self.to_rating_key(user, song),
			self.to_rating_key(ne, song),
			self.to_time_key(user, song),
			self.to_time_key(ne, song),
			decoder=util.float_decoder)
		logger.debug('Retrieved ratings, and timestamps')
		return (u_rating, ne_rating), (u_time, ne_time)

	def get_random_taste(self, n: int = 100) -> np.ndarray:
		logger.info(
			'Randomly retrieving the music taste from one of the first '
			f'{n} users')
		i, stop, taste = 0, self._rng.integers(n), None
		for t in self._redis.scan_iter(match=self.to_taste_key('*')):
			if i > stop:
				break
			else:
				taste = t
				i += 1
		if taste is None:
			raise KeyError('Unable to find the taste of any users')
		else:
			return util.float_array(mp.unpackb(taste))

	def get_features(
			self, *names: str, songs: bool) -> Tuple[np.ndarray, np.ndarray]:
		"""Returns the feature vectors of one or more users or songs."""
		logger.debug(f'Retrieving the features of {names}')
		if songs:
			keys = (self.to_song_key(n) for n in names)
		else:
			keys = (self.to_taste_key(n) for n in names)
		features = self.get(*keys, decoder=mp.unpackb)
		if missing := [f for f in features if f is None]:
			logger.warning(
				f'Unable to find features for {len(missing)} of {len(names)} '
				f'items. Filtering out missing features. The following are '
				f'the items with missing features: {missing}')
		present = [(n, f) for n, f in zip(names, features) if f is not None]
		names = np.array([n for n, _ in present])
		features = util.float_array([f for _, f in present])
		return names, features

	def set_features(
			self,
			names: Iterable[str],
			values: Iterable[np.ndarray],
			song: bool,
			expire: int = None) -> bool:
		if song:
			keys = (self.to_song_key(n) for n in names)
		else:
			keys = (self.to_taste_key(n) for n in names)
		return self.set(keys, values, expire=expire, encoder=mp.packb)

	def get_neighbors(self, user: User, units: str = 'mi') -> np.ndarray:
		"""Returns the other nearby users of a given user."""
		logger.info(f'Finding the neighbors of user {user.name}')
		if all(loc := (user.long, user.lat, user.rad)):
			ne = self._redis.georadius(self.get_location_key(), *loc, units)
			logger.info(f'Found {len(ne)} neighbors of user {user.name}')
		else:
			logger.warning(f'Unable to find the location of user {user.name}')
			ne = []
		return np.array([n.decode('utf-8') for n in ne])

	def get_user(self, name: str) -> User:
		"""Returns a user with all associated attributes populated"""
		logger.info(f'Retrieving attributes of user {name}')
		taste = self.get_features(name, songs=False)[1][0]
		bias_and_radius = (self.to_bias_key(name), self.to_radius_key(name))
		bias, radius = self.get(*bias_and_radius, decoder=util.float_decoder)
		values = (taste, bias, radius)
		if all(missing := [v is None for v in values]):
			raise KeyError(f'Unable to find user {name}')
		if any(missing):
			keys = [self.to_taste_key(name), *bias_and_radius]
			logger.warning(
				f'Unable to find all attributes of user {name}: '
				f'{[k for k, v in zip(keys, values) if v is None]}')
		if loc := self.get_location(name):
			long, lat = loc[0]
			long, lat, float(long), float(lat)
		else:
			logger.warning(f'Unable to find the location of user {name}')
			long, lat = None, None
		return User(
			name=name, taste=taste, bias=bias, long=long, lat=lat, rad=radius)

	@classmethod
	def get_clusters_time_key(cls) -> str:
		return 'ctime'

	@classmethod
	def to_cluster_key(cls, cluster: Union[int, str]) -> str:
		return f'cluster:{cluster}'

	@classmethod
	def get_location_key(cls):
		return 'location'

	@classmethod
	def to_radius_key(cls, user: str) -> str:
		return f'rad:{user}'

	@classmethod
	def to_scores_key(cls, user: str, ne: str) -> str:
		return f'cscores:{user}.{ne}'

	@classmethod
	def to_ratings_key(cls, user: str, ne: str, cluster: str) -> str:
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
