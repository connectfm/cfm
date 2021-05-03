import datetime
import logging
import numpy as np
from scipy.spatial import distance
from typing import Callable, Union

DAY_IN_SECS = 86_400
NOW = datetime.datetime.utcnow()


def get_logger(
		name: str = None,
		level: Union[str, int] = logging.INFO) -> logging.Logger:
	if logging.getLogger(name).hasHandlers():
		log = logging.getLogger(name)
		log.setLevel(level)
	else:
		logging.basicConfig(format='%(levelname)s: %(message)s', level=level)
		log = logging.getLogger(name)
	return log


logger = get_logger(__name__)


def similarity(x: np.ndarray, y: np.ndarray, metric: Callable) -> np.ndarray:
	if np.ndim(x) == 1:
		x = np.array([x])
		dists = distance.cdist(x, y, metric=metric)[0]
	else:
		dists = distance.cdist(x, y, metric=metric)
	return 1 / (1 + dists)


def float_array(arr):
	return np.array(arr, dtype=np.float32)


def if_none(value, none):
	return none if value is None else value


def float_decoder(b: bytes) -> float:
	return float(b.decode('utf-8'))


def delta(timestamp: float) -> float:
	"""Returns the difference in days between a timestamp and now."""
	logger.debug(f'Computing time delta of timestamp {timestamp}')
	diff = NOW - datetime.datetime.utcfromtimestamp(timestamp)
	diff = diff.total_seconds() / DAY_IN_SECS
	logger.debug(f'Computed time delta (days): {diff}')
	return diff
