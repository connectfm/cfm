import json
import redis
import re
import msgpack_numpy as mp # Serializing numpy arrays to store in Redis
import numpy as np
import logging

# Per AWS documentation - AWS apparently creates its own logging instance with its own metadata
# Not doing this will lead to some logs not being written.
if logging.getLogger().hasHandlers():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
else:
    logger = logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logger = logging.getLogger()


# For ensuring ordering of features
FEATURE_LOOKUP = {
                   "danceability":0,
                   "energy":1,
                   "loudness":2,
                   "speechiness":3,
                   "acousticness":4,
                   "instrumentalness":5,
                   "liveness":6,
                   "valence":7,
                   "tempo":8,
                   "duration":9
                  }

DYNAMO = "aws:dynamodb"
TABLE_GROUP = ".*table/(.+)/stream.*"
# AWS Elasticache data
ELASTICACHE = {
    'PORT': 6379,
    # 'HOST': 'localhost'
    'HOST': "demo2-ro.wrnqpf.ng.0001.use1.cache.amazonaws.com"
}

# obj should be json.loads(event)
def update_redis(obj: dict) -> None: #(async)
    # Establish connection to redis
    conn = redis.Redis(host=ELASTICACHE["HOST"], port=ELASTICACHE["PORT"])
    logger.info("Connected to Redis DB")

    # Loop through each record
    for record in obj["Records"]:
        table = re.search(TABLE_GROUP, record["eventSourceARN"])
        table = table.group(1) # Extracts the name of the dynamodb table from the ARN

        logger.info(f"Reading record from table={table}")
        logger.debug(f"Record: {record}")

        if record["eventSource"] == DYNAMO:
            # Remove each of the item's keys from Redis
            if record["eventName"] == "REMOVE":
                remove_attributes(conn, prepare_for_redis(record, table))

            # Add each of the item's keys from Redis
            elif record["eventName"] == "INSERT" or record["eventName"] == "MODIFY":
                update_attributes(conn, prepare_for_redis(record, table))

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
def update_attributes(conn, update: dict) -> None: #(async)
    for key, value in update.items():
        if key.lower() == "location":
            for id, coords in value.items():
                res = conn.geoadd("location", *coords, int(id)) # Coords are 100% float by now, id is string
                logger.info(f"Added location id:{id}, value:{coords}, response code={res}")
        else:
            res = conn.set(key, value)
            logger.info(f"Added key id:{key}, value:{value}, response code={res}")


# Same as update_attribute, except it deletes items from redis
def remove_attributes(redis, update: dict) -> None:
    for key, value in update.items():
        if key.lower() == "location":
            for id, coords in value.items():
                res = redis.zrem("location", int(id))
                logger.info(f"Deleted location {id}, response code={res}")
        else:
            res = redis.delete(key) # Note that this returns '0' if the key didn't exist
            logger.info(f"Deleted key {key}, response code={res}")


# Takes an attribute from an item and extracts the type and value of the attribute in Python types
# This handles all types from a Record, it's likely we won't run into a lot of them.
def val_loads(attr):
    type = (list(attr.keys())[0]).upper()

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
    features = [] * len(FEATURE_LOOKUP)
    for attr in img:
        if attr != "Id":
            value = val_loads(img[attr])
            # Hard coded max values for Loudness & Tempo
            if attr == "Loudness": # Max of 0, range: [-60, 0]
                value = min(0, value)
            elif attr == "Tempo": # Max of 150, range: [50, 150]
                value = min(150, value)
            features[FEATURE_LOOKUP[attr]] = value
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
    except:
        new_img = None
    try:
        old_img = dyn_changes["OldImage"]
    except:
        old_img = None

    img = new_img
    # Choose an image to iterate through based on table update type
    if record["eventName"] == "REMOVE":
        img = old_img

    # If the table type is Song, then we can just handle it right away and return our feature vector
    if table == "Song":
        return {f"song:{id}" : build_feature_array(img)}

    # Grab the id now since we skip it in the image itself
    id = int(val_loads(img["Id"]))
    print(6)
    # Initialize an empty Location field because we'll populate it with ids
    items["Location"] = {}
    for attr in img:
        if attr.lower() != "id":
            value = val_loads(img[attr])
            attr = attr.capitalize()

            if attr in ("Bias", "Radius"):
                items[f"{attr}:{id}"] = value

            elif attr == "Location":
                items["Location"][id] = value # Should already be [float, float]

            elif attr == "Rating": # TODO: How are we getting SONG ID with current Dynamo Schema?
                items[f"{attr}:{id}:<SONG-ID?>"] = value

            elif attr == "Timestamp": # TODO: Same as above^^
                items[f"time:{id}:<SONG-ID?>"] = value

            elif attr == "Taste":
                items[f"taste:{id}"] = build_feature_array(img) # TODO: This will not work, but can use a similar setup for extracting features

    return items

# For testing -- remove later on.
if __name__ == '__main__':
    with open("sample_res.json", 'r') as f:
        update_redis(json.load(f))
