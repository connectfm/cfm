import numpy as np

rng = np.random.default_rng()


def get_coordinate(n: int = 1) -> np.ndarray:
	"""Randomly generates n longitude-latitude coordinate pairs"""
	lat = rng.uniform(-90., 90., size=n)
	long = rng.uniform(-180., 180., size=n)
	coordinates = np.vstack(lat, long).T
	return coordinates


def get_taste(n: int = 1, d: int = 10) -> np.ndarray:
	"""Randomly generates n d-dimensional taste vectors"""
	return rng.standard_normal((n, d))
