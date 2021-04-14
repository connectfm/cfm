import numbers
from typing import Iterable, Union

import attr

Integers = Union[int, Iterable[int]]
Reals = Union[numbers.Real, Iterable[numbers.Real]]
Strings = Union[str, Iterable[str]]


@attr.s(slots=True, frozen=True)
class ConnectDB:
	"""
	API
	4. Compute taste-song distance (key = userID_songID)
	5. Compute taste-taste similarity (key = songID_songID)
	"""

	def get_bias(self, user: Strings) -> Reals:
		pass

	def get_neighbors(self, user: Strings) -> Iterable[str]:
		pass

	def get_rating(self, user: str, *songs: str) -> Integers:
		pass

	def get_cluster(self, song: Strings) -> Strings:
		pass

	def get_user_similarity(self, user: str, *users: str) -> Reals:
		pass

	def get_user_song_dissimilarity(self, user: str, *songs: str) -> Reals:
		pass
