import attr
import codecs
import json
import msgpack_numpy as mp
import numpy as np
import pickle
import random
import redis
from numbers import Real
from scipy.spatial import distance
from typing import (
	Any, Callable, Dict, Iterable, NoReturn, Optional, Sequence, Tuple, Union
)

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
	max_score_caches = attr.ib(type=int, default=1)
	n_rating_caches = attr.ib(type=int, default=1)
	min_similarity = attr.ib(type=int, default=None)
	metric = attr.ib(type=Callable, default=distance.euclidean)
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
			lats: Union[Real, Iterable[Real]]) -> NoReturn:
		self._redis.geoadd(self.get_location_key(), *zip(longs, lats, names))

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

	def cache_scores(
			self,
			user: str,
			ne: str,
			value: Iterable[Real]) -> bool:
		"""Caches the cluster scores of a user.

		The data structure of the cached cluster scores is a mapping from a
		user to timestamped cluster scores:

			{<user>: {"value": <scores>, "time": <timestamp>}}

		The timestamp indicates when the cluster scores were computed.

		When caching scores, there will always be between 1 and
		max_score_caches in the data structure. When there are
		max_score_caches in the data structure, the features of all users are
		used to determine which user to remove. To maximize the probability
		that a neighbor will be at least min_similarity similar to any one of
		the cached entries, the entry that is most similar to all other
		entries is removed. This guarantees the maximum variance in the
		cached user features.

		Returns:
			True if the value was cached successfully, and False otherwise.
		"""
		logger.info(f'Caching cluster scores of user {user} and neighbor {ne}')
		cached = util.if_none(self.get_cached_scores(user, ne), {})
		# Guaranteed at least 1 entry
		cached[ne] = {'value': tuple(value), 'time': util.NOW.timestamp()}
		if len(cached) == self.max_score_caches:
			# Avoid self-bias having the user also be a neighbor
			others = np.array([c for c in cached if c != user])
			users = np.insert(others, 0, user)
			users, tastes = self.get_features(*users, no_none=True, song=False)
			if user == users[0]:
				similarity = util.similarity(tastes, tastes, self.metric)
				aggregate = np.sum(similarity, axis=0)
				# Keep the user in case of no neighbors
				cached.pop(users[np.argmax(aggregate[1:]) + 1])
			else:
				remove = self._rng.choice(others)
				logger.warning(
					f'Unable to find taste of user {user}. Removing {remove} '
					f'from the cached entries')
				cached.pop(remove)
		key = self.to_scores_key(user)
		updated = self.set(key, cached, encoder=json.dumps)
		self._log_cache_event(updated, key)
		return updated

	def get_cached_scores(
			self,
			user: str,
			ne: str,
			*,
			fuzzy: bool = False) -> Union[Dict, np.ndarray, None]:
		"""Attempts to get the cached cluster scores for a user.

		Args:
			user: User of the cluster scores.
			ne: Reference neighbor when finding a fuzzy match to scores. Not
				used when fuzzy is False.
			fuzzy: True will try to find the cluster scores whose reference
				neighbor is at least min_similarity similar to neighbor ne
				and more similar than all other cached entries.
		Returns:
			If fuzzy is False, an optional dictionary of cached entries that
			are still valid, based on their timestamp. Otherwise,
			an optional numpy array of the cluster scores corresponding to
			the neighbor that is most similar to the reference neighbor ne.
		"""
		if scores := self.get(self.to_scores_key(user), decoder=json.loads)[0]:
			is_valid = self._is_valid
			# We may not have any valid scores
			scores = {k: v for k, v in scores.items() if is_valid(v['time'])}
			if fuzzy:
				# Neighbor could also be the user
				if ne in scores:
					scores = scores[ne]['value']
				elif (scores := self._fuzzy(user, ne, scores)) is None:
					logger.warning('No valid cached cluster scores exist')
		else:
			logger.warning(
				f'Unable to find cached cluster scores for user {user}')
		return scores

	def _fuzzy(self, user: str, ne: str, scores: Dict) -> Optional[np.ndarray]:
		others, tastes, scores = self._filter(user, ne, scores)
		if scores := scores if scores else None:
			ne_taste, o_tastes = tastes
			similarity = util.similarity(ne_taste, o_tastes, self.metric)
			if any(close_enough := self.min_similarity < similarity):
				idx = np.flatnonzero(close_enough)
				most_similar = others[np.argmax(similarity[idx])]
				scores = util.float_array(scores[most_similar]['value'])
		return scores

	def _filter(
			self,
			user: str,
			ne: str,
			scores: Dict) -> Tuple[np.ndarray, Tuple, Dict]:
		users = np.array([ne, *(s for s in scores if s != user)])
		users, tastes = self.get_features(*users, song=False)
		# Users may just contain neighbor because scores is empty
		if len(users) == 1:
			others = np.array([ne])
			ne_taste, o_tastes = tastes[0], np.array([tastes[0]])
		else:
			others, ne_taste, o_tastes = users[1:], tastes[0], tastes[1:]
		if ne_taste is None:
			logger.warning(
				f'Unable to find the taste of neighbor {ne}. Returning None '
				f'for the cluster scores')
			scores = None
		elif all(missing := np.isnan(o_tastes)):
			logger.warning(
				f'Unable to find any tastes for users. Returning None for the '
				f'cluster scores. The following users have missing tastes: '
				f'{others}')
			scores = None
		else:
			logger.warning(
				f'Unable to find the tastes of all users. Removing missing '
				f'tastes. The following users have missing tastes: '
				f'{others[np.flatnonzero(missing)]}')
			present = np.flatnonzero(~missing)
			others, o_tastes = others[present], o_tastes[present]
			scores = {p: scores[p] for p in present}
		return others, (ne_taste, o_tastes), scores

	def cache_ratings(
			self,
			user: str,
			ne: str,
			cluster: str,
			value: np.ndarray):
		# Same idea as caching cluster scores
		# Get the cached ratings and filter invalid
		# If not at max, add
		# Otherwise, add and remove the least similar one
		pass

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
		if data := self.get(key, decoder=json.loads)[0]:
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
			self,
			*names: str,
			song: bool,
			no_none: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		"""Returns the feature vectors of one or more users or songs."""
		logger.debug(f'Retrieving the features of {names}')
		if song:
			keys = (self.to_song_key(n) for n in names)
		else:
			keys = (self.to_taste_key(n) for n in names)
		feats = self.get(*keys, decoder=mp.unpackb)
		if missing := [f for f in feats if f is None] and no_none:
			logger.warning(
				f'Unable to find features for {len(missing)} of {len(names)} '
				f'items. Filtering out missing features. The following are '
				f'the items with missing features: {missing}')
			present = [(n, f) for n, f in zip(names, feats) if f is not None]
		else:
			present = [(n, f) for n, f in zip(names, feats)]
		names = np.array([n for n, _ in present])
		feats = util.float_array([f for _, f in present])
		return names, feats

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
		taste = self.get_features(name, song=False)[1][0]
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
	def to_scores_key(cls, user: str) -> str:
		return f'cscores:{user}'

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
