import jsonpickle
import logging
import os
from typing import Any, Dict

import model
import recommend
import util

# Environment variables
MIN_SIMILAR = int(os.getenv('MIN_SIMILAR', default=0.2))
MAX_SCORES = int(os.getenv('MAX_SCORES', default=10))
MAX_RATINGS = int(os.getenv('MAX_RATINGS', default=10))
CACHE = bool(os.getenv('CACHE', default=False))
N_SONGS = int(os.getenv('N_SONGS', default=1_000))
N_NEIGHBORS = int(os.getenv('N_NEIGHBORS', default=100))
METRIC = os.getenv('METRIC', default='euclidean').lower()
SEED = os.getenv('SEED')
if (LOG_LEVEL := os.getenv('LOG_LEVEL', default=logging.INFO)) is not None:
	LOG_LEVEL = LOG_LEVEL.capitalize()
# Globals
logger = util.get_logger(__name__, LOG_LEVEL)


def handle(event, context):
	logger.info(f'## ENV VARS\n{jsonpickle.encode(dict(**os.environ))}')
	logger.info(f'## EVENT\n{(event := jsonpickle.encode(event))}')
	logger.info(f'## CONTEXT\n{jsonpickle.encode(context)}')
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
