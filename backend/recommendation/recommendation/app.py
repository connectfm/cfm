import jsonpickle
import os
from typing import Any, Callable, Dict, Optional

import model
import recommend
import util


def _convert(
		val: Optional[Any], converter: Callable[[Any], Any]) -> Optional[Any]:
	return val if val is None else converter(val)


# Environment variables
MIN_SIMILAR = _convert(os.getenv('MIN_SIMILAR', default=0.2), float)
MAX_SCORES = _convert(os.getenv('MAX_SCORES', default=10), int)
MAX_RATINGS = _convert(os.getenv('MAX_RATINGS', default=10), int)
CACHE = _convert(os.getenv('CACHE', default=False), bool)
N_SONGS = _convert(os.getenv('N_SONGS', default=1_000), int)
N_NEIGHBORS = _convert(os.getenv('N_NEIGHBORS', default=100), int)
METRIC = _convert(os.getenv('METRIC', default='euclidean'), str.lower)
SEED = os.getenv('SEED')
LOG_LEVEL = _convert(os.getenv('LOG_LEVEL', default='INFO'), str.capitalize)
# Globals
logger = util.get_logger(__name__, LOG_LEVEL)


def handle(event, context):
	logger.info('## ENV VARS\n%s', jsonpickle.encode(dict(**os.environ)))
	logger.info('## EVENT\n%s', event := jsonpickle.encode(event))
	logger.info('## CONTEXT\n%s', jsonpickle.encode(context))
	try:
		recommendation = _handle(event['body'])
		response = _response(200, recommendation, event)
	except KeyError as e:
		response = _response(400, repr(e), event)
	return response


def _handle(user: str) -> str:
	with model.RecommendDB(
			min_similar=MIN_SIMILAR,
			max_ratings=MAX_RATINGS,
			max_scores=MAX_SCORES,
			seed=SEED) as db:
		rec = recommend.Recommender(
			db=db,
			metric=METRIC,
			cache=CACHE,
			n_songs=N_SONGS,
			n_neighbors=N_NEIGHBORS,
			seed=SEED)
		return rec.recommend(user)


def _response(status: int, message: Any, event: Any) -> Dict:
	return {
		'statusCode': status,
		'headers': {'x-custom-header': 'connectfm-recommendation'},
		'body': {
			'message': message,
			'input': event,
		}
	}
