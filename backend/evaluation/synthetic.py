import uuid
from typing import Any, Union

import attr
import numpy as np

from context import model


@attr.s(slots=True)
class RecommendData:
	"""Synthetic data generation for connect.fm recommendation evaluation"""
	seed = attr.ib(type=Any, default=None)
	_rng = attr.ib(type=np.random.Generator, init=False, repr=False)

	def __attrs_post_init__(self):
		self._rng = np.random.default_rng(self.seed)

	def get_users(self, n: int = 1, d: int = 10) -> np.ndarray:
		"""Randomly generates n users d-dimensional taste vectors

		All other User attributes are populated as well.
		"""
		names = (uuid.uuid4() for _ in range(n))
		biases = self.get_biases(n)
		radii = self.get_radii(n)
		coords = self.get_coordinates(n)
		tastes = self.get_tastes(n, d)
		return np.array([
			model.User(name=n, bias=b, rad=r, lat=la, long=lo, taste=t)
			for n, b, r, la, lo, t in zip(names, biases, radii, coords, tastes)
		])

	def get_coordinates(self, n: int = 1) -> np.ndarray:
		"""Randomly generates n longitude-latitude coordinate pairs"""
		lat = self._rng.uniform(-90., 90., size=n)
		long = self._rng.uniform(-180., 180., size=n)
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
