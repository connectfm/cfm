import attr
import codecs
import functools
import json
import msgpack_numpy as mp
import numpy as np
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
	max_rating_caches = attr.ib(type=int, default=1)
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

	def cache(
			self,
			value: Union[Sequence[Sequence[Real]], Sequence[Real]],
			*,
			user: str,
			ne: str,
			cluster: str = None) -> bool:
		"""Caches the cluster scores (or ratings) of a user (and cluster).

		If caching cluster scores, the data structure is dict from a
		formatted user key to timestamped cluster scores:

			{<user>: {"value": <scores>, "time": <timestamp>}}

		If caching cluster ratings, the key is formatted to contain cluster
		information as <user>.<cluster>. The mapping is otherwise the same.
		The timestamp indicates when the cluster scores were computed.

		When caching, there will always be between 1 and the set maximum
		number of entries. When there are the maximum number of entries,
		the features of all users are used to determine which entry to remove.
		To maximize the probability that a neighbor will be at least
		min_similarity similar to any one of the cached entries, the entry
		that is most similar to all other entries is removed. This guarantees
		the maximum variance in the cached user features. For cached ratings,
		the ratings belonging to the oldest cached cluster are removed.

		Args:
			value: Cluster scores or ratings to cache.
			user: User associated with the scores or ratings.
			ne: Reference neighbor to determine similarity.
			cluster: Cluster associated with the ratings.

		Returns:
			True if the value was cached successfully, and False otherwise.
		"""

		def pop(__cached: Dict, __user: str) -> NoReturn:
			if cluster is None:
				__cached.pop(__user)
			else:
				matched = (k for k in __cached if __user in k)
				times = ((k, __cached[k]['time']) for k in matched)
				oldest, _ = min(times, key=lambda x: x[1])
				__cached.pop(oldest)

		if cluster is None:
			logger.info(
				f'Caching cluster scores of user {user} and neighbor {ne}')
			key = self._cached_scores_key
			name = self._cached_scores_name
			max_entries = self.max_score_caches
		else:
			logger.info(
				f'Caching cluster ratings of user {user}, neighbor {ne} and '
				f'cluster {cluster}')
			key = functools.partial(
				lambda n: self._cached_ratings_key(n, cluster))
			name = self._cached_ratings_name
			max_entries = self.max_rating_caches
		cached = util.if_none(self.get_cached(user, ne, cluster), {})
		cached[key(ne)] = {'value': value, 'time': util.NOW.timestamp()}
		if len(cached) == max_entries:
			# Avoid duplicate entry of the user, if they are also the neighbor
			others = np.array([name(c) for c in cached if user not in c])
			users = np.insert(others, 0, user)
			users, tastes = self.get_features(*users, no_none=True, song=False)
			# User may have been removed if None
			if user == users[0]:
				sim_matrix = util.similarity(tastes, tastes, self.metric)
				cumulative = np.sum(sim_matrix, axis=0)
				# Keep the user in case of no neighbors
				remove = users[np.argmax(cumulative[1:]) + 1]
			else:
				remove = self._rng.choice(others)
				logger.warning(
					f'Unable to find taste of user {user}. Removing {remove} '
					f'from the cached entries')
			pop(cached, remove)
		key = self.to_scores_key(user)
		updated = self.set(key, cached, encoder=json.dumps)
		self._log_cache_event(updated, key)
		return updated

	@staticmethod
	def _cached_ratings_key(name: str, cluster: str) -> str:
		return f'{name}.{cluster}'

	@staticmethod
	def _cached_ratings_name(key: str) -> str:
		return key.split('.')[0]

	@staticmethod
	def _cached_scores_key(name: str) -> str:
		return name

	@staticmethod
	def _cached_scores_name(key: str) -> str:
		return key

	def get_cached(
			self,
			user: str,
			ne: str,
			cluster: str = None,
			*,
			fuzzy: bool = False) -> Optional[Union[Dict, np.ndarray]]:
		"""Attempts to get the cached cluster scores or ratings of a user.

		Args:
			user: User associated with the scores or ratings.
			ne: Reference neighbor to determine similarity.
			cluster: Cluster associated with the ratings.
			fuzzy: True will try to find the cluster values whose reference
				neighbor is at least min_similarity similar to the neighbor
				and more similar than all other cached entries.

		Returns:
			If fuzzy is False, an optional dictionary of cached entries that
			are still valid, based on their timestamp. Otherwise,
			an optional numpy array of the cluster values corresponding to
			the neighbor that is most similar to the reference neighbor.
		"""
		if cluster is None:
			key = self.to_scores_key(user)
			to_key = self._cached_scores_key
			log_value = 'scores'
		else:
			key = self.to_ratings_key(user)
			to_key = functools.partial(
				lambda n: self._cached_ratings_key(n, cluster))
			log_value = 'ratings'
		logger.info(f'Retrieving cached cluster {log_value} for user {user}')
		if cached := self.get(key, decoder=json.loads)[0]:
			is_valid = self._is_valid
			cached = {k: v for k, v in cached.items() if is_valid(v['time'])}
			if fuzzy:
				if (key := to_key(ne)) in cached:
					cached = cached[key]['value']
				else:
					cached = self._fuzzy(user, ne, cluster, cached)
					if cached is None:
						logger.warning('No valid cached values exist')
		else:
			logger.warning(
				f'Unable to find cached {log_value} for user {user}')
		return cached

	def _fuzzy(
			self,
			user: str,
			ne: str,
			cluster: str,
			cached: Dict) -> Optional[np.ndarray]:
		"""Attempts to find the cluster values of the most similar neighbor.

		Args:
			user: User of the cached values.
			ne: Reference neighbor to determine similarity.
			cached: Valid cached values.

		Returns:
			A numpy of the cached values, or None if no valid entries exist.
		"""
		logger.debug(f'Finding a fuzzy cached value to neighbor {ne}')
		others, tastes, cached = self._filter(cached, user, ne, cluster)
		if cached:
			ne_taste, o_tastes = tastes
			similarity = util.similarity(ne_taste, o_tastes, self.metric)
			close_enough = self.min_similarity < similarity
			idx = np.flatnonzero(close_enough)
			logger.debug(f'Found {len(idx)} fuzzy matches')
			if any(close_enough):
				most_similar = others[np.argmax(similarity[idx])]
				if cluster is None:
					most_similar = self.to_scores_key(most_similar)
				else:
					matched = (k for k in cached if most_similar in k)
					times = ((k, cached[k]['time']) for k in matched)
					most_similar, _ = max(times, key=lambda x: x[1])
					most_similar = self.to_ratings_key(most_similar)
				cached = util.float_array(cached[most_similar]['value'])
				logger.debug(f'Most similar fuzzy match: {cached}')
		return cached

	def _filter(
			self,
			cached: Dict,
			user: str,
			ne: str,
			cluster: str = None) -> Tuple[np.ndarray, Tuple, Dict]:
		"""Gets the tastes and filters out missing entries.

		It is possible that cached is empty. In which case, the neighbor is
		also used as the "other" users that are not the user or neighbor,
		as well as the tastes.

		Args:
			user: User of the cached values.
			ne: Reference neighbor to determine similarity.
			cached: Valid cached values.

		Returns:
			A numpy array of "other" users; a tuple where the first entry is
			the neighbor taste and the second entry are the "other" tastes;
			and an optional dict of cached cluster values.
		"""
		if cluster is None:
			key = self._cached_scores_key
			name = self._cached_scores_name
			log_value = 'scores'
		else:
			key = functools.partial(
				lambda n: self._cached_ratings_key(n, cluster))
			name = self._cached_ratings_name
			log_value = 'ratings'
		logger.debug(f'Filtering missing tastes and cached {log_value}')
		users = (name(ne), *(name(ne) for s in cached if user not in s))
		users = np.array(users)
		users, tastes = self.get_features(*users, song=False)
		# Users may just contain neighbor because scores is empty
		if len(users) == 1:
			others = np.array([ne])
			ne_taste, o_tastes = tastes[0], np.array([tastes[0]])
		else:
			others, ne_taste, o_tastes = users[1:], tastes[0], tastes[1:]
		if ne_taste is None:
			logger.warning(f'Unable to find the taste of neighbor {ne}')
			cached = None
		elif all(missing := np.isnan(o_tastes)):
			logger.warning(
				f'Unable to find any tastes for users. The following users '
				f'have missing tastes: {others}')
			cached = None
		else:
			logger.warning(
				f'Unable to find the tastes of all users. Removing missing '
				f'tastes. The following users have missing tastes: '
				f'{others[np.flatnonzero(missing)]}')
			present = np.flatnonzero(~missing)
			others, o_tastes = others[present], o_tastes[present]
			cached = {key(o): cached[key(o)] for o in others}
		return others, (ne_taste, o_tastes), cached

	@staticmethod
	def _log_cache_event(result: bool, name: str, expire: int = None):
		if result:
			if expire is None:
				logger.info(f'Successfully cached {name} without expiration')
			else:
				logger.info(f'Successfully cached {name} for {expire} seconds')
		else:
			logger.warning(f'Failed to cache {name}')

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
	def to_ratings_key(cls, user: str) -> str:
		return f'cratings:{user}'

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
