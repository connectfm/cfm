import attr
import codecs
import datetime
import functools
import itertools
import json
import numpy as np
import random
import re
import redis
from numbers import Real
from scipy.spatial import distance
from typing import (
	Any, AnyStr, Callable, Dict, Iterable, NoReturn, Optional, Sequence,
	Tuple, Union)

import util

logger = util.get_logger(__name__)
# Type aliases
_Strings = Union[str, Iterable[str]]
_Values = Union[Any, Iterable[Any]]
_Reals = Union[Real, Iterable[Real]]
_OptReal = Optional[Real]
_SequenceOpt = Sequence[Optional[Any]]
_Coder = Union[Callable[[Any], Any], str]
_Metric = Callable[[Any, Any], Real]
_Biases = Sequence[_OptReal]
_Clusters = Optional[Iterable[str]]
_NumClusters = Optional[int]
_Radii = Sequence[_OptReal]
_Locations = Sequence[Optional[Tuple[Real, Real]]]
_Funcs = Tuple[Callable[[str], str], Callable[[str], bool]]
_RatingsTimes = Tuple[Tuple[_OptReal, _OptReal], Tuple[_OptReal, _OptReal]]
# Globals
_FEATURE_DECODER = json.loads
_FEATURE_ENCODER = json.dumps
_CACHE_DECODER = json.loads
_CACHE_ENCODER = json.dumps
_LIST_ENCODER = json.dumps
_LIST_DECODER = json.loads
_MAX_REDIS_SET = 524_287


# noinspection PyUnresolvedReferences
@attr.s(slots=True)
class User:
	"""A user of connect.fm.

	Attributes:
		name: A string indicating the name of the user.
		taste: A numpy ndarray of real numbers indicating the user's taste
			(default: None).
		bias: A real number indicting the user's recommendation bias
			(default: None).
		lat: A real number indicating the user's latitude (default: None).
		long: A real number indicating the user's longitude (default: None).
		rad: A real number indicating the user's radius (default: None).
	"""
	name = attr.ib(type=str, kw_only=True)
	taste = attr.ib(type=np.ndarray, default=None, kw_only=True)
	bias = attr.ib(type=Real, default=None, kw_only=True)
	lat = attr.ib(type=Real, default=None, kw_only=True)
	long = attr.ib(type=Real, default=None, kw_only=True)
	rad = attr.ib(type=Real, default=None, kw_only=True)


# noinspection PyUnresolvedReferences
@attr.s(slots=True)
class RecommendDB:
	"""Data access object for the connect.fm recommender system database.

	Attributes:
		max_scores: An integer indicating the number of cluster scores to
			cache per user. A cluster score is the probability of it being
			sampled and is based on the songs within it (default: 1).
		max_ratings: An integer indicating the number of cluster ratings to
			cache per user. This is the number of clusters, not the number of
			ratings per cluster (default: 1).
		min_similar: A float indicating the minimum similarity between a
			sampled a neighbor and another neighbor whose data is already
			cached for the recommendation to be based on the cached neighbor's
			data (default: None).
		seed: Any value to use when setting the random seed (default: None).
		metric: A callable distance function that expects two inputs and
			outputs a value that represents the distance between them. This is
			used to compute similarity using the following transformation:
			similarity = 1 / (1 + distance) (default: Euclidean).
		config: A dictionary of configuration parameters used to establish
			connection to the database (default: None).
	"""
	max_scores = attr.ib(type=int, default=1, kw_only=True)
	max_ratings = attr.ib(type=int, default=1, kw_only=True)
	min_similar = attr.ib(type=Real, default=0, kw_only=True)
	seed = attr.ib(type=Any, default=None, kw_only=True)
	metric = attr.ib(type=_Metric, default=distance.euclidean, kw_only=True)
	config = attr.ib(type=Dict, default=None, kw_only=True, repr=False)
	_rng = attr.ib(type=np.random.Generator, init=False, repr=False)
	_redis = attr.ib(type=redis.Redis, init=False, repr=False)

	def __attrs_post_init__(self):
		random.seed(self.seed)
		self._rng = np.random.default_rng(self.seed)
		logger.debug('Seed: %s', self.seed)

	def __enter__(self):
		logger.debug('Establishing connection to the database')
		config = {} if self.config is None else self.config
		self._redis = redis.Redis(**config)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		logger.debug('Closing connection to the database')
		self._redis.close()

	def get(self, *keys: str, decoder: _Coder = None) -> _SequenceOpt:
		"""Gets one or more values corresponding to the keys.

		Args:
			*keys: String keys for each object to retrieve.
			decoder: A string or a single-argument callable to decode the byte
				value into the desired object. If a string, it is assumed to
				be one of the supported codecs by the codecs built-in library.
		Returns:
			A tuple of objects, in the order specified by the keys. If an
			object is not found, None is returned as its value.
		"""
		return tuple(
			v if v is None else self._decode(v, decoder)
			for v in self._redis.mget(*keys))

	def set(
			self,
			keys: _Strings,
			values: _Values,
			*,
			expire: int = None,
			encoder: _Coder = None) -> bool:
		"""Sets one or more key-value pairs.

		Args:
			keys: A string or iterable of strings associated with the values.
			values: Any value or iterable associated with the keys.
			expire: An integer indicating the number of seconds until which
				all key-value pairs expire.
			encoder: A string or a single-argument callable to encode the value
				into string, byte, or number type. If a string, it is
				assumed to be one of the supported codecs by the codecs
				built-in library.

		Returns:
			True if all key-value pairs were stored successfully, and False
			otherwise.
		"""

		def _chunk(data: Dict) -> Iterable[Dict]:
			# https://github.com/StackExchange/StackExchange.Redis/issues/201
			it = iter(data)
			return (
				{k: data[k] for k in itertools.islice(it, _MAX_REDIS_SET)}
				for _ in range(0, len(data), _MAX_REDIS_SET))

		if isinstance(keys, str):
			items = {keys: self._encode(values, encoder)}
		else:
			items = {k: self._encode(v, encoder) for k, v in zip(keys, values)}
		pipe = self._redis.pipeline()
		for chunk in _chunk(items):
			pipe.mset(chunk)
			if expire is not None:
				for k in chunk:
					pipe.expire(k, expire)
		return all(pipe.execute())

	def get_bias(self, *names: str) -> _Biases:
		"""Gets the bias of all users with the specified names."""
		return self.get(*(self.to_bias_key(n) for n in names))

	def set_bias(
			self,
			names: _Strings,
			values: _Values,
			*,
			expire: int = None) -> bool:
		"""Sets the bias of all users specified with the specified values.

		Args:
			names: A string or iterable of strings used as the lookup key.
			values: A real number or iterable of real numbers indicating the
				bias of each user.
			expire: An integer indicating the number of seconds until which
				all key-value pairs expire.

		Returns:
			True if all biases were stored successfully, and False otherwise.
		"""
		if isinstance(names, str):
			return self.set(self.to_bias_key(names), values, expire=expire)
		keys = (self.to_radius_key(n) for n in names)
		return self.set(keys, values, expire=expire)

	def get_radius(self, *names: str) -> _Radii:
		"""Returns the radius of all users with the specified names."""
		return self.get(*(self.to_radius_key(n) for n in names))

	def set_radius(
			self,
			names: _Strings,
			values: _Values,
			expire: int = None) -> bool:
		"""Sets the radius of all users specified with the specified values.

		Args:
			names: A string or iterable of strings used as the lookup key.
			values: A real number or iterable of real numbers indicating the
				radius of each user.
			expire: An integer indicating the number of seconds until which
				all key-value pairs expire.

		Returns:
			True if all radii were stored successfully, and False otherwise.
		"""
		keys = (self.to_radius_key(n) for n in names)
		return self.set(keys, values, expire=expire)

	def get_location(self, *names: str) -> _Locations:
		"""Gets the locations of all users with the specified names.

		Returns:
			A sequence of pairs, where each pair is (longitude, latitude).
		"""
		return self._redis.geopos(self.get_location_key(), *names)

	def set_location(
			self, names: _Strings, longs: _Reals, lats: _Reals) -> NoReturn:
		"""Sets the radius of all users specified with the specified values."""
		locs = itertools.chain(*zip(longs, lats, names))
		self._redis.geoadd(self.get_location_key(), *locs)

	def get_clusters(self) -> _Clusters:
		"""Gets the names of all clusters."""
		clusters = self.get(self.get_clusters_key(), decoder=_LIST_DECODER)[0]
		return (self._get_name(c) for c in clusters)

	def set_clusters(
			self,
			names: _Strings,
			songs: Sequence[_Strings],
			*,
			replace: bool = True) -> bool:
		"""Sets the clusters, number of clusters, and clusters timestamp.

		All set operations are done as one transaction, so if at least one
		set operation fails, all of them do.

		Args:
			names: A string or iterable of strings indicating the cluster
				names.
			songs: A sequence of strings or sequence of iterables of strings
				indicating the songs to add to each cluster.
			replace: If True, deletes the contents of the clusters, if they
				already exist. Otherwise, the songs will be added to an
				existing cluster.

		Returns:
			True if all set operations succeed and false otherwise.
		"""

		def wrap(val, test) -> np.ndarray:
			if isinstance(test, str):
				wrapped = (val,)
			else:
				wrapped = tuple(val)
			return np.array(wrapped)

		names, songs = wrap(names, names), wrap(songs, songs[0])
		keys = np.array([self.to_cluster_key(n) for n in names])
		pipe = self._redis.pipeline()
		if replace:
			pipe.delete(*keys)
		for k, c_songs in zip(keys, songs):
			pipe.sadd(k, *(self.to_song_key(s) for s in c_songs))
		keys = keys.tolist()
		pipe.mset({
			self.get_clusters_key(): self._encode(keys, encoder=_LIST_ENCODER),
			self.get_num_clusters_key(): len(songs),
			self.get_clusters_time_key(): datetime.datetime.now().timestamp()})
		return all(pipe.execute())

	def get_num_clusters(self) -> _NumClusters:
		"""Gets the number of clusters."""
		key = self.get_num_clusters_key()
		if (n_clusters := self.get(key, decoder='utf-8')[0]) is not None:
			n_clusters = int(n_clusters)
		return n_clusters

	def get_clusters_time(self) -> float:
		"""Gets the time at which the clusters were created as a timestamp."""
		key = self.get_clusters_time_key()
		return self.get(key, decoder=util.float_decoder)[0]

	def get_songs(self, cluster: str, n: int = None) -> Iterable[str]:
		"""Gets the songs of a cluster.

		Returns:
			If n is specified, then an iterable of at most n songs. Otherwise,
			an iterable of all the songs in the cluster.
		"""
		key = self.to_cluster_key(cluster)
		name = self._get_name
		if n is None:
			songs = (name(s) for s in self._redis.sscan_iter(key))
		else:
			songs = (name(s) for s in self._redis.srandmember(key, n))
		return songs

	def get_ratings(self, user: str, ne: str, song: str) -> _RatingsTimes:
		"""Gets the ratings and times between a user, neighbor, and song.

		Returns:
			The structure of the returned tuple is:
			(user_rating, neighbor_rating), (user_time, neighbor_time)
		"""
		logger.debug(
			'Retrieving ratings, and timestamps for user %s and neighbor %s',
			user, ne)
		u_rating, ne_rating, u_time, ne_time = self.get(
			self.to_rating_key(user, song),
			self.to_rating_key(ne, song),
			self.to_time_key(user, song),
			self.to_time_key(ne, song),
			decoder=util.float_decoder)
		return (u_rating, ne_rating), (u_time, ne_time)

	def get_random_taste(self) -> Optional[np.ndarray]:
		"""Gets a random taste.

		Raises:
			KeyError: If unable to find the taste of any users.

		Returns:
			If a taste was found, a numpy ndarray.
		"""
		logger.info('Randomly retrieving a music taste')
		tastes = self._redis.scan_iter(match=self.to_taste_key('*'))
		taste = None
		for t in zip(range(1), tastes):
			taste = t
		if taste is None:
			raise KeyError('Unable to find the taste of any users')
		else:
			taste = self._decode(taste, decoder=_FEATURE_DECODER)
		return util.float_array(taste)

	def get_features(
			self,
			*names: str,
			song: bool,
			no_none: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		"""Gets the feature vectors of one or more users or songs.

		Attributes:
			*names: String names of users or songs.
			song: If True, gets song features. Otherwise, gets user features.
				If names is users and song is True, then it is very likely no
				features will be found, and vice versa.
			no_none: If True, only returns the features that are not None.
				Note that this implies the ordering of the names and return
				values may differ.
		Returns:
			A tuple (names, features). If no_none is True, only the names
			whose features are not None are returned. Otherwise, they are the
			same as given.
		"""
		logger.debug('Retrieving the features of %s', repr(names))
		names = np.array(names)
		key = self.to_song_key if song else self.to_taste_key
		feats = self.get(*(key(n) for n in names), decoder=_FEATURE_DECODER)
		if any(missing := np.array([f is None for f in feats])) and no_none:
			idx = np.flatnonzero(missing)
			logger.warning(
				'Unable to find features for %d of %d items. Filtering out '
				'missing features. The following are the items with missing '
				'features: %s', len(missing), len(names), names[idx])
		present = np.flatnonzero(~missing)
		names, feats = names[present], util.float_array(feats)[present]
		return names, feats

	def set_features(
			self,
			names: _Strings,
			values: _Reals,
			*,
			song: bool,
			expire: int = None) -> bool:
		"""Sets the features for one or more users or songs.

		Note that numpy arrays will be converted to lists to ensure they are
		serializable.

		Args:
			names: A string or iterable of strings indicating the names of
				users or songs.
			values: A real number or iterable of real numbers indicating the
				features of each user or song.
			song: True (False) indicates the given names and features
				correspond to songs (ref. users).
			expire: If None, the entries will never expire. Otherwise,
				they will be removed after the specified number of seconds.

		Returns:
			True if the setting of features was successful and False otherwise.
		"""
		key = self.to_song_key if song else self.to_taste_key
		if isinstance(names, str):
			names = (names,)
			values = (values,)
		keys = (key(n) for n in names)
		vals = (v.tolist() if isinstance(v, np.ndarray) else v for v in values)
		return self.set(keys, vals, expire=expire, encoder=_FEATURE_ENCODER)

	def get_neighbors(
			self,
			user: Union[str, User],
			*,
			long: Real = None,
			lat: Real = None,
			rad: Real = None,
			n: int = None,
			units: str = 'mi') -> np.ndarray:
		"""Gets the names of other nearby users (neighbors) of a given user.

		Args:
			user: A string or User object indicating the user of interest. If
				a User, the its attribute values of long, lat, and rad will be
				used.
			long: A real number indicating the longitude of the user.
			lat: A real number indicating the latitude of the user.
			rad: A real number indicating the radius of the user.
			n: An integer indicating the maximum number of neighbors to return.
				If None, all neighbors within the radius are returned.
			units: The units of the radius.

		Returns:
			A numpy ndarray of strings indicating the names of all neighbors.
		"""
		if isinstance(user, User):
			user, long, lat, rad = user.name, user.long, user.lat, user.rad
		logger.info('Finding the neighbors of user %s', user)
		if all(loc := (long, lat, rad)):
			key = self.get_location_key()
			ne = self._redis.georadius(key, *loc, units, count=n)
			ne = [u for n in ne if (u := self._decode(n, 'utf-8')) != user]
			logger.info('Found %d neighbors of user %s', len(ne), user)
		else:
			logger.warning('Unable to find the location of user %s', user)
			ne = []
		return np.array(ne)

	def get_user(self, name: str) -> User:
		"""Returns a user with all associated attributes populated."""
		logger.info('Retrieving attributes of user %s', name)
		keys = np.array((
			self.to_taste_key(name),
			self.to_bias_key(name),
			self.to_radius_key(name)))
		taste = self.get_features(name, song=False)[1][0]
		bias, radius = self.get(*keys[1:], decoder=util.float_decoder)
		values = (taste, bias, radius)
		if all(missing := np.array([v is None for v in values])):
			raise KeyError(f'Unable to find user %s', name)
		if any(missing):
			args = (name, keys[np.flatnonzero(missing)])
			logger.warning(
				'Unable to find all attributes of user %s: %s', *args)
		if loc := self.get_location(name):
			long, lat = loc[0]
			long, lat, float(long), float(lat)
		else:
			logger.warning('Unable to find the location of user %s', name)
			long, lat = None, None
		return User(
			name=name, taste=taste, bias=bias, long=long, lat=lat, rad=radius)

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

			{user: {ne: {"value": <scores>, "time": <timestamp>},...}

		If caching cluster ratings, the key is formatted to contain cluster
		information as ne.cluster. The mapping is otherwise the same.
		The timestamp indicates when the cluster scores were computed.

		When caching, there will always be between 1 and the set maximum
		number of entries. When there are the maximum number of entries,
		the features of all users are used to determine which entry to remove.
		To maximize the probability that a neighbor will be at least
		min_similarity similar to any one of the cached entries, the cached
		neighbor that is most similar to all other neighbors is removed. This
		guarantees the maximum variance in the cached neighbor features. For
		cached ratings, the ratings belonging to the oldest cached cluster are
		removed.

		Args:
			value: Any serializable value.
			user: A string indicating the user associated with the scores or
				ratings.
			ne: A string indicating the neighbor associated with the cached
				value.
			cluster: A string indicating the name of the cluster associated
				with the ratings.

		Returns:
			True if the value was cached successfully, and False otherwise.
		"""
		self._log_cache_start(user, ne, cluster)
		max_caches = self.max_scores if cluster is None else self.max_ratings
		db_key, cache_key = self._keys(user, ne, cluster)
		cached = util.if_none(self.get_cached(user, ne, cluster), {})
		cache = {
			'value': tuple(value),
			'time': datetime.datetime.utcnow().timestamp()}
		cached[cache_key] = cache
		if len(cached) > max_caches:
			self._evict(cached, user, cluster)
		updated = self.set(db_key, cached, encoder=_CACHE_ENCODER)
		self._log_cache_end(updated, db_key)
		return updated

	def get_cached(
			self,
			user: str,
			ne: str,
			cluster: str = None,
			*,
			fuzzy: bool = False) -> Optional[Union[Dict, np.ndarray]]:
		"""Attempts to get the cached cluster scores or ratings of a user.

		Args:
			user: A string indicating the user associated with the scores or
				ratings.
			ne: A string indicating the neighbor associated with the cached
				value.
			cluster: A string indicating the name of the cluster associated
				with the ratings.
			fuzzy: True will try to find the cluster values whose associated
				neighbor is at least min_similarity similar to the neighbor
				and more similar than all other cached entries.

		Returns:
			If fuzzy is False, an optional dictionary of cached entries that
			are still valid, based on their timestamp. Otherwise,
			an optional numpy ndarray of the cluster values corresponding to
			the neighbor that is most similar to the given neighbor.
		"""
		self._log_retrieval(user, ne, cluster)
		db_key, cache_key = self._keys(user, ne, cluster)
		if cached := self.get(db_key, decoder=_CACHE_DECODER)[0]:
			valid = self._valid
			cached = {k: v for k, v in cached.items() if valid(v['time'])}
		if cached and fuzzy:
			if cache_key in cached:
				cached = cached[cache_key]['value']
			elif (cached := self._fuzzy(cached, user, ne, cluster)) is None:
				logger.warning(
					'No valid cached %s exist', self._log_value(cluster))
		else:
			self._log_failure(user, ne, cluster)
		return cached

	def _evict(self, cached: Dict, user: str, cluster: str = None):
		"""Removes the most similar neighbor in the cache."""
		to_name, is_user = self._funcs(user, cluster)
		others = np.array([to_name(c) for c in cached if not is_user(c)])
		users = np.insert(others, 0, user)
		users, tastes = self.get_features(*users, no_none=True, song=False)
		if len(users) > 0 and user == users[0]:
			sim_matrix = util.similarity(tastes, tastes, self.metric)
			cumulative = np.sum(sim_matrix, axis=0)
			most_similar = users[np.argmax(cumulative[1:]) + 1]
			_, is_most_similar = self._funcs(most_similar, cluster)
			self._pop(cached, is_most_similar)
		else:
			logger.warning('Unable to find the taste of user %s', user)
			if len(others) > 0:
				remove = self._rng.choice(others)
				_, is_remove = self._funcs(remove, cluster)
				logger.warning('Removing %s from cached entries', remove)
				self._pop(cached, is_remove)

	def _keys(
			self, user: str, ne: str, cluster: str = None) -> Tuple[str, str]:
		"""Gets the database and cache keys.

		Attributes:
			user: A string indicating the user associated with the scores or
				ratings.
			ne: A string indicating the neighbor associated with the cached
				value.
			cluster: A string indicating the name of the cluster associated
				with the ratings.

		Returns:
			A tuple strings with the form (database_key, cache_key).
		"""
		if cluster is None:
			db_key = self.to_scores_key(user)
			cache_key = self.to_scores_key(ne)
		else:
			db_key = self.to_ratings_key(user)
			cache_key = self.to_ratings_key(ne, cluster)
		return db_key, cache_key

	def _fuzzy(
			self,
			cached: Dict,
			user: str,
			ne: str,
			cluster: str = None) -> Optional[np.ndarray]:
		"""Attempts to find the cluster values of the most similar neighbor.

		Args:
			cached: A dictionary of valid cached values.
			user: A string indicating the user associated with the scores or
				ratings.
			ne: A string indicating the neighbor associated with the cached
				value.
			cluster: A string indicating the name of the cluster associated
				with the ratings.

		Returns:
			A numpy ndarray the cached values, or None if no valid entries
			exist.
		"""
		logger.info('Finding a fuzzy cached value to neighbor %s', ne)
		others, tastes, cached = self._filter(cached, user, ne, cluster)
		if not cached:
			return None
		ne_taste, o_tastes = tastes
		similarity = util.similarity(ne_taste, o_tastes, self.metric)
		idx = np.flatnonzero(close_enough := self.min_similar < similarity)
		logger.info('Found %d fuzzy matches', len(idx))
		if any(close_enough):
			closest = others[(idx := np.argmax(similarity[idx]))]
			if cluster is None:
				key = self.to_scores_key(closest)
			else:
				_, is_user = self._funcs(closest, cluster)
				matched = (k for k in cached if is_user(k))
				times = ((k, cached[k]['time']) for k in matched)
				key, _ = max(times, key=lambda x: x[1])
			closest = cached[key]['value']
			logger.info(
				'Most similar fuzzy match (similarity): %s (%f)',
				user, round(similarity[idx], 6))
		else:
			closest = None
		return closest

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
			cached: A dictionary of valid cached values.
			user: A string indicating the user associated with the scores or
				ratings.
			ne: A string indicating the neighbor associated with the cached
				value.
			cluster: A string indicating the name of the cluster associated
				with the ratings.

		Returns:
			A numpy array of "other" users; a tuple where the first entry is
			the neighbor taste and the second entry are the "other" tastes;
			and an optional dict of cached cluster values.
		"""
		logger.info(
			'Filtering missing tastes and cached %s', self._log_value(cluster))
		to_name, is_user = self._funcs(user, cluster)
		names = {ne, *(to_name(k) for k in cached if not is_user(k))}
		users, tastes = self.get_features(*names, no_none=True, song=False)
		others, ne_taste, o_tastes = (None,) * 3
		if len(users) == 0 or users[0] != ne:
			cached = None
		elif len(users) == 1:
			others = np.array([ne])
			ne_taste, o_tastes = tastes[0], np.array([tastes[0]])
		else:
			others, ne_taste, o_tastes = users[1:], tastes[0], tastes[1:]
		if len(names) != len(users):
			missing = repr(names.difference(users))
			logger.warning('Unable to find tastes of some users: %s', missing)
		return others, (ne_taste, o_tastes), cached

	def _funcs(self, user: str, cluster: str = None) -> _Funcs:
		"""Returns a tuple of callables (to_name(str), is_user(str))."""
		if cluster is None:
			to_name = self.from_scores_key
			reg = re.compile(fr'{self.to_scores_key(user)}+')
		else:
			to_name = functools.partial(lambda k: self.from_ratings_key(k)[0])
			reg = re.compile(fr'{self.to_ratings_key(user, cluster)}*')
		is_user = functools.partial(lambda k: reg.match(k) is not None)
		return to_name, is_user

	def _valid(self, timestamp: float) -> bool:
		"""Checks if a given timestamp is valid with respect to caching.

		Returns:
			True if the timestamp is valid and False otherwise. If the
			cluster time is missing, the timestamp is assumed valid.
		"""
		if c_time := self.get_clusters_time():
			valid = timestamp > c_time
		else:
			logger.warning(
				'Unable to find the cluster scores timestamp. Assuming '
				'the associated value is still representative')
			valid = True
		return valid

	@classmethod
	def get_num_clusters_key(cls) -> str:
		"""Gets the key associated with the number of clusters."""
		return 'nclusters'

	@classmethod
	def get_clusters_key(cls) -> str:
		"""Gets the key associated with all of the cluster names."""
		return 'clusters'

	@classmethod
	def get_clusters_time_key(cls) -> str:
		"""Returns the key associated with the cluster created-at time."""
		return 'ctime'

	@classmethod
	def to_cluster_key(cls, cluster: str) -> str:
		"""Returns the key associated with the cluster's."""
		return f'cluster:{cluster}'

	@classmethod
	def get_location_key(cls):
		"""Returns the key associated with all users' locations."""
		return 'location'

	@classmethod
	def to_radius_key(cls, user: str) -> str:
		"""Returns the key associated with the user's radius."""
		return f'rad:{user}'

	@classmethod
	def to_scores_key(cls, user: str) -> str:
		"""Returns the key associated with the user's cluster scores."""
		return f'scores:{user}'

	@classmethod
	def from_scores_key(cls, key: str) -> str:
		"""Returns the user associated with the scores key."""
		return key.split(':')[-1]

	@classmethod
	def to_ratings_key(cls, user: str, cluster: str = None) -> str:
		"""Returns the key associated with user (and cluster) ratings"""
		if cluster is None:
			key = f'ratings:{user}'
		else:
			key = f'ratings:{user}.{cluster}'
		return key

	@classmethod
	def from_ratings_key(cls, key: str) -> Sequence[str]:
		"""Returns the user (and cluster) associated with the ratings key."""
		return tuple(key.split(':')[-1].split('.'))

	@classmethod
	def to_bias_key(cls, user: str) -> str:
		"""Returns the key associated with the user's bias."""
		return f'bias:{user}'

	@classmethod
	def to_taste_key(cls, user: str) -> str:
		"""Returns the key associated with the user's taste."""
		return f'taste:{user}'

	@classmethod
	def to_song_key(cls, song: str) -> str:
		"""Returns the key associated with the song's features."""
		return f'song:{song}'

	@classmethod
	def to_rating_key(cls, user: str, song: str) -> str:
		"""Returns the key associated with the user's song rating."""
		return f'rating:{user}.{song}'

	@classmethod
	def to_time_key(cls, user: str, song: str) -> str:
		"""Returns the key associated with the user's song rating timestamp."""
		return f'time:{user}.{song}'

	@staticmethod
	def _decode(value: Any, decoder: _Coder) -> Any:
		"""Decodes the value with the given decoder."""
		if isinstance(decoder, str):
			value = codecs.decode(value, decoder)
		elif isinstance(decoder, Callable):
			value = decoder(value)
		return value

	@staticmethod
	def _encode(value: Any, encoder: _Coder) -> Any:
		"""Encodes the value with the given encoder."""
		if isinstance(encoder, str):
			value = codecs.encode(value, encoder)
		elif isinstance(encoder, Callable):
			value = encoder(value)
		return value

	@staticmethod
	def _get_name(key: AnyStr) -> str:
		"""Parses the key to get the name component."""
		if isinstance(key, bytes):
			key = key.decode('utf-8')
		return key.split(':')[-1]

	@staticmethod
	def _pop(cached: Dict, is_user: Callable[[str], bool]):
		"""Removes the oldest entry of the user from the cache."""
		matched = ((k, v['time']) for k, v in cached.items() if is_user(k))
		oldest, _ = min(matched, key=lambda x: x[1])
		cached.pop(oldest)

	@staticmethod
	def _log_value(cluster: str = None) -> str:
		return 'scores' if cluster is None else 'ratings'

	# noinspection PyTypeChecker
	@staticmethod
	def _log_cache_end(
			result: bool,
			name: str,
			expire: int = None) -> NoReturn:
		if result:
			if expire is None:
				logger.info('Successfully cached %s without expiration', name)
			else:
				logger.info(
					f'Successfully cached %s for %d seconds', name, expire)
		else:
			logger.warning('Failed to cache %s', name)

	# noinspection PyTypeChecker
	@staticmethod
	def _log_failure(user: str, ne: str, cluster: str = None) -> NoReturn:
		if cluster is None:
			logger.warning('Unable to find cached scores for user %s', user)
		else:
			logger.warning(
				'Unable to find cached ratings for user %s, neighbor %s, '
				'and cluster %s', user, ne, cluster)

	# noinspection PyTypeChecker
	@staticmethod
	def _log_retrieval(user: str, ne: str, cluster: str = None) -> NoReturn:
		if cluster is None:
			logger.info(
				'Retrieving cached cluster scores for user %s and neighbor %s',
				user, ne)
		else:
			logger.info(
				'Retrieving cached cluster ratings for user %s, and neighbor '
				'%s, and cluster %s', user, ne, cluster)

	# noinspection PyTypeChecker
	@staticmethod
	def _log_cache_start(user: str, ne: str, cluster: str = None) -> NoReturn:
		if cluster is None:
			logger.info(
				'Caching cluster scores of user %s and neighbor %s', user, ne)
		else:
			logger.info(
				'Caching cluster ratings of user %s, neighbor %s and, cluster '
				'%s', user, ne, cluster)
