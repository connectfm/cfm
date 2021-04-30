import attr
import codecs
import functools
import itertools
import json
import msgpack_numpy as mp
import numpy as np
import random
import re
import redis
from numbers import Real
from scipy.spatial import distance
from typing import (
	Any, Callable, Dict, Iterable, List, NoReturn, Optional, Sequence, Tuple,
	Union
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
	max_scores = attr.ib(type=int, default=1)
	max_ratings = attr.ib(type=int, default=1)
	min_similarity = attr.ib(type=int, default=0)
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
		locs = itertools.chain(*zip(longs, lats, names))
		self._redis.geoadd(self.get_location_key(), *locs)

	def get_clusters(self) -> Iterable[str]:
		keys = self._redis.scan_iter(match=self.to_cluster_key('*'))
		return (self._get_name(k) for k in keys)

	@staticmethod
	def _get_name(encoded: bytes) -> str:
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
			value: Any,
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
		if cluster is None:
			logger.info(
				f'Caching cluster scores of user {user} and neighbor {ne}')
			max_caches = self.max_scores
		else:
			logger.info(
				f'Caching cluster ratings of user {user}, neighbor {ne} and, '
				f'cluster {cluster}')
			max_caches = self.max_ratings
		db_key, cache_key = self._keys(user, ne, cluster)
		cached = util.if_none(self.get_cached(user, ne, cluster), {})
		cached[cache_key] = {'value': value, 'time': util.NOW.timestamp()}
		if len(cached) > max_caches:
			self._evict(cached, user, cluster)
		updated = self.set(db_key, cached, encoder=json.dumps)
		self._log_cache_event(updated, db_key)
		return updated

	def _evict(self, cached: Dict, user: str, cluster: str = None):
		to_name, is_user = self._funcs(user, cluster)
		# Avoid duplicate entry of the user, if they are also the neighbor
		others = np.array([to_name(c) for c in cached if not is_user(c)])
		users = np.insert(others, 0, user)
		users, tastes = self.get_features(*users, no_none=True, song=False)
		# User may have been removed if None
		if len(users) > 0 and user == users[0]:
			sim_matrix = util.similarity(tastes, tastes, self.metric)
			cumulative = np.sum(sim_matrix, axis=0)
			# Keep the user in case of no neighbors
			most_similar = users[np.argmax(cumulative[1:]) + 1]
			_, cache_key = self._keys(ne=most_similar, cluster=cluster)
			self._pop(cached, most_similar, cache_key)
		else:
			logger.warning(f'Unable to find the taste of user {user}')
			if len(others) > 0:
				remove = self._rng.choice(others)
				_, cache_key = self._keys(ne=remove, cluster=cluster)
				logger.warning(f'Removing {remove} from cached entries')
				self._pop(cached, remove, cache_key)

	def _pop(self, cached: Dict, user: str, key: str, cluster: str = None):
		if cluster is None:
			cached.pop(key)
		else:
			_, is_user = self._funcs(user, cluster)
			matched = (k for k in cached if is_user(key))
			times = ((k, cached[k]['time']) for k in matched)
			oldest, _ = min(times, key=lambda x: x[1])
			cached.pop(oldest)

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
		self._log_retrieval(user, ne, cluster)
		db_key, cache_key = self._keys(user, ne, cluster)
		if cached := self.get(db_key, decoder=json.loads)[0]:
			valid = self._valid
			cached = {k: v for k, v in cached.items() if valid(v['time'])}
		if cached and fuzzy:
			if cache_key in cached:
				cached = cached[cache_key]['value']
			elif (cached := self._fuzzy(cached, user, ne, cluster)) is None:
				log_value = 'scores' if cluster is None else 'ratings'
				logger.warning(f'No valid cached {log_value} exist')
		else:
			self._log_failure(user, ne, cluster)
		return cached

	@staticmethod
	def _log_retrieval(user: str, ne: str, cluster: str = None):
		if cluster is None:
			logger.info(
				f'Retrieving cached cluster scores for user {user} and '
				f'neighbor {ne}')
		else:
			logger.info(
				f'Retrieving cached cluster ratings for user {user}, and '
				f'neighbor {ne}, and cluster {cluster}')

	def _keys(self, user: str = None, ne: str = None, cluster: str = None):
		if cluster is None:
			db_key = self.to_scores_key(user)
			cache_key = self.to_scores_key(ne)
		else:
			db_key = self.to_ratings_key(user)
			cache_key = self.to_ratings_key(ne, cluster)
		return db_key, cache_key

	@staticmethod
	def _log_failure(user: str, ne: str, cluster: str = None):
		if cluster is None:
			logger.warning(f'Unable to find cached scores for user {user}')
		else:
			logger.warning(
				f'Unable to find cached ratings for user {user}, neighbor '
				f'{ne}, and cluster {cluster}')

	def _fuzzy(
			self,
			cached: Dict,
			user: str,
			ne: str,
			cluster: str = None) -> Optional[np.ndarray]:
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
			closest = self._find_closest(cached, others, tastes, cluster)
		else:
			closest = None
		return closest

	def _find_closest(
			self,
			cached: Dict,
			others: np.ndarray,
			tastes: Tuple[np.ndarray, np.ndarray],
			cluster: str = None) -> Optional[np.ndarray]:
		ne_taste, o_tastes = tastes
		similarity = util.similarity(ne_taste, o_tastes, self.metric)
		idx = np.flatnonzero(close_enough := self.min_similarity < similarity)
		logger.debug(f'Found {len(idx)} fuzzy matches')
		if any(close_enough):
			closest = others[np.argmax(similarity[idx])]
			closest = self._closest_key(cached, closest, cluster)
			closest = cached[closest]['value']
			logger.debug(f'Most similar fuzzy match: {cached}')
		else:
			closest = None
		return closest

	def _closest_key(
			self, cached: Dict, closest: str, cluster: str = None) -> str:
		if cluster is None:
			key = self.to_scores_key(closest)
		else:
			to_name, is_user = self._funcs(closest, cluster)
			matched = (to_name(k) for k in cached if is_user(k))
			times = ((k, cached[k]['time']) for k in matched)
			newest, _ = max(times, key=lambda x: x[1])
			key = self.to_ratings_key(newest, cluster)
		return key

	def _filter(
			self,
			cached: Dict,
			user: str,
			ne: str,
			cluster: str = None) -> Tuple:
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
		log_value = 'scores' if cluster is None else 'ratings'
		logger.debug(f'Filtering missing tastes and cached {log_value}')
		to_name, is_user = self._funcs(user, cluster)
		names = {ne, *(to_name(k) for k in cached if not is_user(k))}
		users, tastes = self.get_features(*names, no_none=True, song=False)
		others, ne_taste, o_tastes = (None,) * 3
		if len(users) == 0 or users[0] != ne or ne not in cached:
			cached = None
		elif len(users) == 1:
			others = np.array([ne])
			ne_taste, o_tastes = tastes[0], np.array([tastes[0]])
		else:
			others, ne_taste, o_tastes = users[1:], tastes[0], tastes[1:]
		if len(names) != len(users):
			missing = names.difference(users)
			logger.warning(f'Unable to find tastes of some users: {missing}')
		return others, (ne_taste, o_tastes), cached

	def _funcs(self, user: str, cluster: str = None) -> Tuple[Callable, ...]:
		if cluster is None:
			to_name = self.from_scores_key
			reg = re.compile(fr'{self.to_scores_key(user)}+')
		else:
			to_name = functools.partial(lambda k: self.from_ratings_key(k)[0])
			reg = re.compile(fr'{self.to_ratings_key(user, cluster)}*')
		not_user = functools.partial(lambda k: reg.match(k) is not None)
		return to_name, not_user

	@staticmethod
	def _log_cache_event(result: bool, name: str, expire: int = None):
		if result:
			if expire is None:
				logger.info(f'Successfully cached {name} without expiration')
			else:
				logger.info(f'Successfully cached {name} for {expire} seconds')
		else:
			logger.warning(f'Failed to cache {name}')

	def _valid(self, timestamp: float) -> bool:
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
			song: str) -> Tuple[Tuple[Real, Real], Tuple[Real, Real]]:
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
		names = np.array(names)
		if song:
			keys = (self.to_song_key(n) for n in names)
		else:
			keys = (self.to_taste_key(n) for n in names)
		feats = self.get(*keys, decoder=mp.unpackb)
		missing = np.array([f is None for f in feats])
		if any(missing) and no_none:
			idx = np.flatnonzero(missing)
			logger.warning(
				f'Unable to find features for {len(missing)} of {len(names)} '
				f'items. Filtering out missing features. The following are '
				f'the items with missing features: {names[idx]}')
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
		logger.info(f'Finding the neighbors of user {(name := user.name)}')
		if all(loc := (user.long, user.lat, user.rad)):
			ne = self._redis.georadius(self.get_location_key(), *loc, units)
			ne = [u for n in ne if (u := n.decode('utf-8')) != name]
			logger.info(f'Found {len(ne)} neighbors of user {name}')
		else:
			logger.warning(f'Unable to find the location of user {name}')
			ne = []
		return np.array(ne)

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
	def to_cluster_key(cls, cluster: str) -> str:
		return f'cluster:{cluster}'

	@classmethod
	def get_location_key(cls):
		return 'location'

	@classmethod
	def to_radius_key(cls, user: str) -> str:
		return f'rad:{user}'

	@classmethod
	def to_scores_key(cls, user: str) -> str:
		return f'scores:{user}'

	@classmethod
	def from_scores_key(cls, key: str) -> str:
		return key.split(':')[-1]

	@classmethod
	def to_ratings_key(cls, user: str, cluster: str = None) -> str:
		if cluster is None:
			key = f'ratings:{user}'
		else:
			key = f'ratings:{user}.{cluster}'
		return key

	@classmethod
	def from_ratings_key(cls, key: str) -> List[str]:
		return key.split(':')[-1].split('.')

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
