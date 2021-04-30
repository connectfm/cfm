import itertools
import json
import os
import random
import time
import uuid
from typing import Any, List, NoReturn, Tuple, Union

import attr
import numpy as np
import pandas as pd

import recommend
import util
from context import model


def preprocess_tracks_dataset(dir_path: str) -> NoReturn:
	"""Removes irrelevant columns and creates features.json"""
	tracks = pd.read_csv(os.path.join(dir_path, 'tracks.csv'))
	drop = ('time_signature', 'artists', 'key', 'explicit')
	tracks.drop(columns=drop, inplace=True)
	tracks.to_csv(os.path.join(dir_path, 'tracks.csv'))
	order = (
		'danceability',
		'energy',
		'loudness',
		'speechiness',
		'acousticness',
		'instrumentalness',
		'liveness',
		'valence',
		'tempo')
	ids = tracks['id'].to_numpy()
	features = tracks[order].to_numpy()
	with open(os.path.join(dir_path, 'features.json'), 'w') as f:
		features = {i: f.tolist() for i, f in zip(ids, features)}
		json.dump(features, f)


def load_features(dir_path: str) -> Tuple[np.ndarray, np.ndarray]:
	with open(os.path.join(dir_path, 'features.json'), 'r') as f:
		data = json.load(f)
		keys = np.array([k for k in data])
		features = util.float_array([data[k] for k in keys])
		return keys, features


def store_users(db: model.RecommendDB, *users: model.User) -> NoReturn:
	names = [u.name for u in users]
	db.set_bias(names, (u.bias for u in users))
	db.set_features(names, (u.taste for u in users), song=False)
	db.set_radius(names, (u.rad for u in users))
	db.set_location(names, (u.long for u in users), (u.lat for u in users))


def store_clusters(db: model.RecommendDB, *clusters: np.ndarray) -> NoReturn:
	for c, songs in enumerate(clusters):
		db.set_cluster(str(c), *songs)
	db.set_clusters_time(time.time())
	db.set_num_clusters(len(clusters))


def store_features(
		db: model.RecommendDB,
		names: np.ndarray,
		features: np.ndarray) -> NoReturn:
	db.set_features(names, features.tolist(), song=True)


@attr.s(slots=True)
class RecommendData:
	"""Synthetic data generation for connect.fm recommendation evaluation"""
	seed = attr.ib(type=Any, default=None)
	_rng = attr.ib(type=np.random.Generator, init=False, repr=False)

	def __attrs_post_init__(self):
		self._rng = np.random.default_rng(self.seed)

	def get_users(
			self,
			n: int = 1,
			d: int = 10,
			use_uuid: bool = False,
			small_world: bool = False) -> np.ndarray:
		"""Randomly generates n users d-dimensional taste vectors

		All other User attributes are populated as well.
		"""
		if use_uuid:
			names = (str(uuid.uuid4()) for _ in range(n))
		else:
			names = (str(i) for i in range(n))
		biases = self.get_biases(n)
		radii = self.get_radii(n)
		coords = self.get_coordinates(n, small_world=small_world)
		tastes = self.get_tastes(n, d)
		return np.array([
			model.User(name=n, bias=b, rad=r, lat=la, long=lo, taste=t)
			for n, b, r, (lo, la), t
			in zip(names, biases, radii, coords, tastes)])

	def get_coordinates(
			self, n: int = 1, small_world: bool = False) -> np.ndarray:
		"""Randomly generates n longitude-latitude coordinate pairs"""
		if small_world:
			long = self._rng.uniform(0, 0.1, size=n)
			lat = self._rng.uniform(0, 0.1, size=n)
		else:
			long = self._rng.uniform(-180, 180, size=n)
			lat = self._rng.uniform(-85, 85, size=n)
		coordinates = np.vstack((long, lat)).T
		return coordinates[0] if n == 1 else coordinates

	def get_tastes(self, n: int = 1, d: int = 10) -> np.ndarray:
		"""Randomly generates n d-dimensional taste vectors"""
		return self._rng.standard_normal((n, d))

	def get_biases(self, n: int = 1) -> Union[np.ndarray, float]:
		"""Randomly generates n biases"""
		uniform = self._rng.uniform
		return uniform(0, 1) if n == 1 else uniform(0, 1, size=(n,))

	def get_radii(self, n: int = 1) -> Union[np.ndarray, float]:
		"""Randomly generates n radii"""
		uniform = self._rng.uniform
		return uniform(0, 100) if n == 1 else uniform(0, 100, size=(n,))

	@staticmethod
	def get_clusters(*items: str, n: int = 5) -> List[np.ndarray]:
		n = min(len(items), n)
		clusters = [list() for _ in range(n)]
		for c, item in zip(itertools.cycle(range(n)), items):
			clusters[c].append(item)
		return [np.array(c) for c in clusters]


def main():
	data = RecommendData()
	keys, features = load_features('data/')
	n_songs = 1_000_000
	keys, features = keys[:n_songs], features[:n_songs]
	users = data.get_users(n_users := 5, d=len(features[0]), small_world=True)
	clusters = data.get_clusters(*keys, n=2)
	with model.RecommendDB(
			min_similar=0.1, max_scores=1_000, max_ratings=10) as db:
		store_features(db, keys, features)
		store_users(db, *users)
		store_clusters(db, *clusters)
		rec = recommend.Recommender(db, n_songs=1_000, cache=False)
		for i in range(50):
			u = random.randint(0, n_users - 1)
			rec.recommend(users[u].name)


if __name__ == '__main__':
	main()
