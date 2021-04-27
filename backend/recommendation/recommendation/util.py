import datetime
import logging
import numpy as np
from typing import Any

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DAY_IN_SECS = 86_400
NOW = datetime.datetime.utcnow()


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
