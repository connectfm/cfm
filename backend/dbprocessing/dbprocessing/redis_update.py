import json
import aioredis
import asyncio
import re
import msgpack_numpy as mp # Serializing numpy arrays to store in Redis
import numpy as np
import logging


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# For ensuring ordering of features
FEATURE_LOOKUP = {
                   "Danceability":0,
                   "Energy":1,
                   "Loudness":2,
                   "Speechiness":3,
                   "Acousticness":4,
                   "Instrumentalness":5,
                   "Liveness":6,
                   "Valence":7,
                   "Tempo":8,
                   "Duration":9
                  }

DYNAMO = "aws:dynamodb"
TABLE_GROUP = ".*table/(\w+)/stream.*"
# AWS Elasticache data
ELASTICACHE = {
    'PORT': 6379,
    'HOST': "connectfm.urj6ir.ng.0001.use2.cache.amazonaws.com"
}

# obj should be json.loads(event)
def update_redis(obj: dict) -> None: #(async)

    # Loop through each record
    for record in obj["Records"]:
        table = re.search(TABLE_GROUP, record["eventSourceARN"])
        table = table.group(1) # Extracts the name of the dynamodb table from the ARN

        logger.info(f"Reading record from DBStream, table={table}")

        if record["eventSource"] == DYNAMO:
            # Remove each of the item's keys from Redis
            if record["eventName"] == "REMOVE":
                remove_attributes(None, prepare_for_redis(record, table))

            # Add each of the item's keys from Redis
            elif record["eventName"] == "INSERT" or record["eventName"] == "MODIFY":
                update_attributes(None, prepare_for_redis(record, table))

            else:
                # If the event name doesn't match, respond with 400
                return {
                    "code": 400,
                    "body": f"Unexpected event_type{event_type}"
                }

    # Return successful result
    return {
        "code": 200,
        "body": "Successfully parsed all db records."
    }


# Accepts the redis connection, key, and attribute for item to update (or add)
# Items should already be properly typecasted via val_loads, so just need to call
#  redis functions here. We also construct the proper key?
def update_attributes(redis, update: dict) -> None: #(async)
    for key, value in update.items():
        if key == "Location":
            for id, coords in value.items():
                # redis.geoadd("location", *coords, int(id)) # Coords are 100% float by now, id is string
                logger.info(f"Added location id:{id}, value:{coords}")
        else:
            # redis.set(key, value)
            logger.info(f"Added key id:{key}, value:{value}")


# Same as update_attribute, except it deletes items from redis
def remove_attributes(redis, update: dict) -> None:
    for key, value in update.items():
        if key == "Location":
            for id, coords in value.items():
                redis.zrem("location", int(id))
                logger.info(f"Deleted location {id}")
        else:
            redis.delete(key) # Note that this returns '0' if the key didn't exist
            logger.info(f"Deleted key {key}")


# Takes an attribute from an item and extracts the type and value of the attribute in Python types
# This handles all types from a Record, it's likely we won't run into a lot of them.
def val_loads(attr):
    type = list(attr.keys())[0]

    # Hardcoded translation of any datatype from DynamoDBStream
    if type == "B": # Blob, binary string
        val = attr[type] # Not much to do here
    elif type == "BOOL": # Boolean
        val = bool(attr[type])
    elif type == "BS": # Blob set, a list of binary strings
        val = attr[type] # Not much to do here
    elif type == "L" or type == "M": # List or Map, potentially a list of maps, etc.
        val = attr[type]
    elif type == "N": # Number - Need to typecast as floats because they start as strings
        val = float(attr[type])
    elif type == "NS": # Number set, should result in just a list of numbers as strings
        val = [float(i) for i in attr[type]]
    elif type == "NULL": # Value is either null or it isn't - we can't extract useful data here
        val =  None
    elif type == "S": # String or String set, both basically the same
        val = attr[type]
    elif type == "SS":
        val = attr[type]

    return val

# Extracts an (ordered) array of features for song/taste items and returns it
def build_feature_array(img: dict):
    features = np.zeros(len(FEATURE_LOOKUP))
    for attr in img:
        if attr != "Id":
            features[FEATURE_LOOKUP[attr]] = val_loads(img[attr])
    return features


# Take a value, create a dictionary of key:value ready for update/remove_attributes
def prepare_for_redis(record: dict, table) -> dict:
    logger.info(f"Prepping record from {table} table")
    # Helper references
    dyn_changes = record["dynamodb"]
    # Dictionary of items to return for Redis population
    items = {}

    try:
        new_img = dyn_changes["NewImage"]
        id = new_img["Id"]["N"]
    except:
        new_img = None
    try:
        old_img = dyn_changes["OldImage"]
        id = old_img["Id"]["N"]
    except:
        old_img = None

    # Choose an image to iterate through based on table update type
    if record["eventName"] == "REMOVE":
        img = old_img
    else:
        img = new_img

    # If the table type is Song, then we can just handle it right away and return our feature vector
    if table == "Song":
        return {f"song:{id}" : mp.packb(build_feature_array(img))}

    # Grab the id now since we skip it in the image itself
    id = img["Id"]["N"]
    # Initialize an empty Location field because we'll populate it with ids
    items["Location"] = {}
    for attr in img:
        if attr != "Id":
            value = val_loads(img[attr])
            attr = attr.capitalize()

            if attr in ("Bias", "Radius"):
                items[f"{attr}:{id}"] = value

            elif attr == "Location":
                items["Location"][id] = value # Should already be [float, float]

            elif attr == "Rating": # TODO: How are we getting SONG ID with current Dynamo Schema?
                items[f"{attr}:{id}:<SONG-ID?>"] = value

            elif attr == "timestamp": # TODO: Same as above^^
                items[f"time:{id}:<SONG-ID?>"] = value

            elif attr == "taste":
                items[f"taste:{id}"] = mp.packb(build_feature_array(img)) # TODO: This will not work, but can use a similar setup for extracting features

    return items

# For testing -- remove later on.
if __name__ == '__main__':
    with open("sample_res.json", 'r') as f:
        update_redis(json.load(f))
