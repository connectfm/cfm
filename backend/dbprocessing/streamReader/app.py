# Entry point for the lambda function
import json
import asyncio
import logging
import datetime

import redis_update as redis

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def handler(event, context):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.info(f"Handling new DBStream at {datetime.utcnow()}")

    # Update redis values based on changes
    try:
        result = loop.run_until_complete(redis.update_redis())
        return result

    except:
        logger.warn("Encountered an unhandled exception in redis.update_redis()")
        return {
            'statusCode': 400,
            'body': json.dumps("Something went wrong!")
        }
