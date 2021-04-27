import datetime
import logging
from typing import Union

import numpy as np

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


def float_array(arr):
	return np.array(arr, dtype=np.float16)


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
