import asyncio
import jsonpickle
import logging
import os

import model
import recommend

MAX_CLUSTERS = int(os.environ['MAX_CLUSTERS'])
if seed := os.environ['SEED'] == 'None':
	SEED = None
else:
	SEED = seed

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def handle(event, context):
	logger.debug(f'## ENV VARS\n{jsonpickle.encode(dict(**os.environ))}')
	logger.debug(f'## EVENT\n{jsonpickle.encode(event)}')
	logger.debug(f'## CONTEXT\n{jsonpickle.encode(context)}')
	recommendation = asyncio.run(_handle(event['body']))
	return {'body': recommendation, 'statusCode': 200}


async def _handle(user: str) -> str:
	async with model.RecommendDB(SEED) as db:
		rec = recommend.Recommender(db, max_clusters=MAX_CLUSTERS, seed=SEED)
		return await rec.recommend(user)
