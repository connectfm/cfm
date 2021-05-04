import numpy as np
import os
import uuid
import logging
import time
from typing import Any, List, NoReturn, Tuple, Union

from context import model, recommend, util
import synthetic


SEED = 12345 # Shout out to Dr. Ray - It'd be helpful for consecutive runs to have similar time for now
np.random.seed(SEED)

# Numbers of users and songs to use in data analysis
N_USERS = 10_000
N_SONGS = 50_000

# Config dictionary for disabling cache in recommender
NO_CACHE = {}


if logging.getLogger().hasHandlers():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
else:
    logger = logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logger = logging.getLogger()


def timed_evaluation(db: model.RecommendDB, 
					 keys: list, 
					 features: list, 
					 users: list,
					 recommender_params: dict = None):
	"""Recommends songs based on the db and hyperparameters specified. 
	Returns the elapsed time to build a recommender object and the recommender object itself.
	"""
	start = time.process_time()

	synthetic.store_features(db, keys, features)
	synthetic.store_users(db, *users)
	synthetic.store_clusters(db, *clusters)

	rec = recommendation.Recommender(db, *recommender_params)

	end = time.process_time()

	return start - end, rec


def recommend_n(user: model.User, 
				rec: recommendation.Recommender, 
				n: int) -> np.ndarray:
	"""Takes a recommender object and user to recommend n songs"""
	return np.array([rec.recommendation(user.name) for _ in range(n)])


def compare_songs_to_user(user: model.User, 
						  song_features: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
	"""Evaluates how similar the recommended songs are in comparison to the user's taste ratings.
	Returns sum of |taste.attr - song.attr| for attr in song over all songs"""
	taste = user.taste
	del_taste = np.zeros(len(taste))

	for features in song_features:
		pass
	
	return del_taste


# For now, just keeping this in a single method, will likely refactor to a class since that makes more sense
def main():
	# Initialize data/references to be used throughout evaluation
	data = synthetic.RecommendData()
	keys, features = load_features('data/')
	keys, features = keys[:N_SONGS], features[:N_SONGS]
	users = data.get_users(N_USERS, d=len(features[0]), small_world=True)
	clusters = data.get_clusters(*keys)

	logger.info(f"Starting new recommendation evaluation with seed {SEED}.")

	logger.info(f"Basic recommendation with max_scores=0, max_ratings=0, min_similar=1 (forced always computing)")
	# Generate a basic recommendation set
	db = model.RecommendDB(max_scores=0, max_ratings=0, min_similar=1, seed=SEED)
	t, rec = time_evaluation(db)
	songs = recommend_n(users[0], rec, 10)
	del_taste = compare_songs_to_user(users[[0], songs])

	logger.info(f"Took {end - start}s\n")

	for i in range(5):
		s = max(np.random.random(), 0.1)

		logger.info(f"Fuzzy matching run #{i+1}, s={s}")
		db = model.RecommendDB(db=0.1, max_ratings=0.1, min_similar=s)
		t, rec = time_evaluation(db)

		logger.info(f"#{i+1} took {t}s\n")
