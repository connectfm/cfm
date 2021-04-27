# Entry point for the lambda function
import sys
import json
import logging
import datetime

import redis_update as redis

# Per AWS documentation - AWS apparently creates its own logging instance with its own metadata
# Not doing this will lead to some logs not being written. 
if logging.getLogger().hasHandlers():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
else:
    logger = logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logger = logging.getLogger()


def handler(event, context):
    logger.info(f"Handling new DBStream at {datetime.datetime.utcnow()}")
    logger.debug(f"Got event: {event}")

    # Update redis values based on changes
    # try:
    result = redis.update_redis(event)
    return result

    # except:
    #     e = sys.exc_info()[0]
    #     logger.warning(f"Ran into an error when attempting to update Redis: {e}")
    #     return {
    #         'statusCode': 400,
    #         'body': json.dumps("Something went wrong!")
    #     }


# For testing -- remove later on.
if __name__ == '__main__':
    with open("sample_res.json", 'r') as f:
        handler(json.load(f), None)
