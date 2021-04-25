import attr
import itertools
import json
import msgpack_numpy as mp
import numpy as np
import os
import pandas as pd
import uuid
from typing import Any, List, NoReturn, Tuple, Union

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
	for u in users:
		db.set(db.to_taste_key(u.name), mp.packb(u.taste))
		db.set(db.to_radius_key(u.name), u.rad)
		db.set(db.to_bias_key(u.name), u.bias)
		db.set_location(u.name, u.long, u.lat)


def store_clusters(db: model.RecommendDB, *clusters: np.ndarray) -> NoReturn:
	for c, songs in enumerate(clusters):
		key = db.to_cluster_key(c)
		db.set_cluster(key, *(db.to_song_key(s) for s in songs))


def store_features(
		db: model.RecommendDB,
		keys: np.ndarray,
		features: np.ndarray) -> NoReturn:
	for k, f in zip(keys, features):
		db.set(db.to_song_key(k), mp.packb(f))


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
			small_world: bool = False) -> np.ndarray:
		"""Randomly generates n users d-dimensional taste vectors

		All other User attributes are populated as well.
		"""
		names = (str(uuid.uuid4()) for _ in range(n))
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
			long = self._rng.uniform(0, 1, size=n)
			lat = self._rng.uniform(0, 1, size=n)
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
	users = data.get_users(10, d=len(features[0]), small_world=True)
	clusters = data.get_clusters(*keys, n=3)
	with model.RecommendDB() as db:
		store_features(db, keys[:100], features[:100])
		store_users(db, *users)
		store_clusters(db, *clusters)
		print(recommend.Recommender(db).recommend(users[0].name))


if __name__ == '__main__':
	main()
