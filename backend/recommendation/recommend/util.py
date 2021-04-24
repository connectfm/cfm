import datetime

import logging
import numpy as np
from typing import Any

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

DAY_IN_SECS = 86_400
NOW = datetime.datetime.utcnow()


def float_array(arr):
	return np.array(arr, dtype=np.float32)


def sample(
		population: np.ndarray,
		weights: np.ndarray,
		with_index: bool = False,
		seed: Any = None) -> Any:
	"""Returns an element from the population using weighted sampling."""
	probs = weights / sum(weights)
	mean = np.round(np.average(probs), 3)
	sd = np.round(np.std(probs), 3)
	logger.debug(f'Mean (sd) probability: {mean} ({sd})')
	rng = np.random.default_rng(seed)
	if with_index:
		population = np.vstack((np.arange(population.size), population)).T
		idx_and_chosen = rng.choice(population, p=probs)
		idx, chosen = idx_and_chosen[0], idx_and_chosen[1:]
		chosen = chosen.item() if chosen.shape == (1,) else chosen
		logger.debug(f'Sampled element (index): {sample} ({idx})')
		chosen = (chosen, idx)
	else:
		chosen = rng.choice(population, p=probs)
		logger.debug(f'Sampled element: {sample}')
	return chosen


def delta(timestamp: float) -> float:
	"""Returns the difference in days between a timestamp and now."""
	logger.debug(f'Computing time delta of timestamp {timestamp}')
	diff = NOW - datetime.datetime.utcfromtimestamp(timestamp)
	diff = diff.total_seconds() / DAY_IN_SECS
	logger.debug(f'Computed time delta (days): {delta}')
	return diff
