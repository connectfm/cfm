import logging
import os
from typing import Any, Dict

import jsonpickle

import model
import recommend
import util

MAX_CLUSTERS = int(os.getenv('MAX_CLUSTERS', default=100))
SEED = os.getenv('SEED')
LOG_LEVEL = os.getenv('LOG_LEVEL', default=logging.INFO)

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
	with model.RecommendDB(SEED) as db:
		rec = recommend.Recommender(db, max_clusters=MAX_CLUSTERS, seed=SEED)
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
