import numpy as np
import os
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
REC_CFG = {
	"n_songs": 1000,
	"n_neighbors": 100,
	"cache": False
}


if logging.getLogger().hasHandlers():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
else:
    logger = logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logger = logging.getLogger()


def timed_evaluation(db: model.RecommendDB,
					 rec_params: dict = None):
	"""Recommends songs based on the db and hyperparameters specified. 
	recommender_params is a dict that should be able to be unpacked into a Recommender init
	Returns the elapsed time to build a recommender object and the recommender object itself.
	"""
	start = time.process_time()

	if rec_params:
		rec = recommend.Recommender(db, **rec_params)
	else:
		rec = recommend.Recommender(db)

	end = time.process_time()

	return end - start, rec


def recommend_n(user: model.User, 
				rec: recommend.Recommender,
				n: int) -> np.ndarray:
	"""Takes a recommender object and user to recommend n songs"""
	return np.array([rec.recommend(user.name) for _ in range(n)])


def compare_songs_to_user(user: model.User, 
						  song_features: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
	"""Evaluates how similar the recommended songs are in comparison to the user's taste ratings.
	Returns sum of |taste.attr - song.attr| for attr in song over all songs.
	NOTE: This is probably more useful for a single-song to observe what sort of songs are being recommended.
	"""
	taste = user.taste
	del_taste = np.zeros(len(taste))

	# Compute the difference between user taste and song features for each song.
	for features in song_features[1]:
		for i in range(len(features)):
			del_taste[i] = abs(taste[i] - features[i])
	
	return del_taste


def dim_reduce_clusters(clusters):
	pass # PCA with scikit learn - do each cluster individually and color each cluster individually


# For now, just keeping this in a single method, will likely refactor to a class since that makes more sense
def main():
	# Initialize data/references to be used throughout evaluation
	data = synthetic.RecommendData()
	keys, features = synthetic.load_features('data/')
	keys, features = keys[:N_SONGS], features[:N_SONGS]
	users = data.get_users(N_USERS, d=len(features[0]), small_world=True)
	clusters = data.get_clusters(*keys)

	dim_reduce_clusters(clusters)

	logger.info(f"Starting new recommendation evaluation with seed {SEED}.")

	logger.info(f"Basic recommendation with max_scores=0, max_ratings=0, min_similar=1 (forced always computing)")
	# Generate a basic recommendation set
	with model.RecommendDB(max_scores=0, max_ratings=0, min_similar=1, seed=SEED) as db:
		# Store preliminary data
		synthetic.store_features(db, keys, features)
		synthetic.store_users(db, *users)
		synthetic.store_clusters(db, *clusters)

		# Track timing for recommendation
		t, rec = timed_evaluation(db, rec_params=REC_CFG)
		songs = recommend_n(users[0], rec, 10)
		songs = db.get_features(*songs, song=True)
		del_taste = compare_songs_to_user(users[0], songs)

		logger.info(f"Of {len(songs)} songs sampled, total dTaste = {del_taste}")

		logger.info(f"Took {t}s\n")

	# Time a few recommendations for fuzzy models
	for i in range(5):
		s = round(max(np.random.random(), 0.1), 3)

		logger.info(f"Fuzzy matching run #{i+1}, s={s}")
		db = model.RecommendDB(max_scores=0.1, max_ratings=0.1, min_similar=s)
		t, rec = timed_evaluation(db)

		logger.info(f"#{i+1} took {t}s\n")

if __name__ == '__main__':
	main()