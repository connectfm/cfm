# Entry point for the lambda function
import sys
import json
import logging
import datetime

import redis_update as redis

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def handler(event, context):
    logger.info(f"Handling new DBStream at {datetime.datetime.utcnow()}")
    logger.debug(f"Got event: {event}")

    # Update redis values based on changes
    try:
        result = redis.update_redis(event)
        return result

    except:
        e = sys.exc_info()[0]
        logger.warn(f"Ran into an error when attempting to update Redis: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps("Something went wrong!")
        }
