import numpy as np
import os
import json
import logging
import time
import datetime
from typing import Any, List, NoReturn, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter
from sklearn import manifold
# import UMAP

from context import model, recommend, util
import synthetic

SEED = 12345
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


def keep_only_songs(keys: np.ndarray,
					features: np.ndarray,
					songs: np.ndarray) -> dict:
	"""Removes songs that haven't been clustered quickly. Returns a dictionary of {songid: feat}"""
	# To keep things fast, load the key and feature pairs into a dict and then remove keys as we see fit.
	full = {}
	to_keep = {}
	for k, f in zip(keys, features):
		full[k] = f

	# Now remove all of the keys not in songs:
	for song in songs:
		try:
			to_keep[song] = full[song]
		except KeyError as e:
			pass

	return to_keep


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
						  song_features: np.ndarray) -> np.ndarray:
	"""Evaluates how similar the recommended songs are in comparison to the user's taste ratings.
	Returns avg of |taste.attr - song.attr| for attr in song over all songs.
	NOTE: This is probably more useful for a single-song to observe what sort of songs are being recommended.
	"""
	taste = user.taste
	del_taste = np.zeros(len(taste))

	# Compute the difference between user taste and song features for each song.
	for features in song_features:
		for i in range(len(features)):
			del_taste[i] = abs(taste[i] - features[i])
	
	return del_taste / len(song_features)


def plot_tsne(clusters: np.ndarray, features: np.ndarray, db: model.RecommendDB):
	"""Plots the original data at the same interval for a few perplexities
	clusters is used for utility and reference.
	It's ideal that songs holds all of the songs in the clusters to make getting features extremely simple"""
	perplexities = [5, 10, 50, 100]
	n_clusters = len(clusters)
	colors = cm.rainbow(np.linspace(0, 1, n_clusters))
	fig, subplots = plt.subplots(2, 2, figsize=(10, 10))
	subplots = subplots.flatten()

	# Compute cluster sizes for reference. Add a 0 to the start to make looping easier
	cluster_sizes = np.array([0] + [len(cluster) for cluster in clusters])

	# Run TSNE for the different perplexities
	for ax, p in zip(subplots, perplexities):
		tsne = manifold.TSNE(n_components=2, perplexity=p, n_iter=1000)
		# Convert the array from x by y by z to x*y by z
		results = tsne.fit_transform(features)

		ax.set_title(f"Perplexity {p}, 1000 iter")

		for i, color in enumerate(colors):
			# We want to plot each cluster as its own color: the first cluster_sizes = 1, the second cluster_sizes = 2, etc.
			ax.scatter(results[cluster_sizes[i]: cluster_sizes[i+1], 0],
					   results[cluster_sizes[i]: cluster_sizes[i+1], 1],
					   color=color, s=8)
			ax.xaxis.set_major_formatter(NullFormatter())
			ax.yaxis.set_major_formatter(NullFormatter())

	plt.show()


def plot_umap(clusters: np.ndarray, db: model.RecommendDB):
	"""Plots the cluters using the UMAP reduction scheme. """
	n_clusters = len(clusters)
	colors = cm.rainbow(np.linspace(0, 1, n_clusters))
	fig, subplots = plt.subplots(2, 2, figsize=(10, 10))
	subplots = subplots.flatten()

	# Extract the features of the clusters and flatten the 5 arrays.
	clusters = np.array([db.get_features(*cluster, song=True)[1] for cluster in clusters])
	cluster_sizes = clusters.shape[1] # Size is a n_clusters * size of a cluster * n_features per song (9)


def test_n_biases(user: model.User, db: model.RecommendDB, rec: recommend.Recommender, n: int) -> np.ndarray:
	"""Changes the bias of a user n times (iter #1 is for a bias of 0).
	Logs the different del_taste values for each bias after recommending 10 songs.
	Returns a matrix. Each vector is an individual del_taste value (difference between taste and song recommended).
	"""
	del_taste = []

	logger.info(f"Sampling for user: {user.name}. features = {db.get_features(user.name, song=False)}")
	songs = recommend_n(user, rec, 10)
	features = db.get_features(*songs, song=True)[1]
	del_taste.append(np.array(compare_songs_to_user(user, features)))
	db.set_bias(user.name, 0)
	user.bias = 0
	# NOTE: It looks like logging a custom level can create some issues when multiple
	# custom levels are specified, so I'm just going to manually print
	# log messages like this to a file.
	with open('eval.log', 'a') as logfile:
		logfile.write(f"Of {len(songs)} songs sampled, avg dTaste = {del_taste[-1]}"
					  f" (sum={sum(del_taste[-1][:-1])}, excluding last term) w/ bias={user.bias}\n")

	for i in range(n):
		# Use FEATURE_LOOKUP to change features and update for this user.
		new_bias = round(max(np.random.random(), 0.1), 3)
		db.set_bias(user.name, new_bias)
		user.bias = new_bias

		songs = recommend_n(user, rec, 10)
		features = db.get_features(*songs, song=True)[1]
		del_taste.append(np.array(compare_songs_to_user(user, features)))

		with open('eval.log', 'a') as logfile:
			logfile.write(f"Of {len(songs)} songs sampled, avg dTaste = {del_taste[-1]}"
						  f" (sum={sum(del_taste[-1][:-1])}, excluding last term) w/ bias={user.bias}\n")
	return np.array(del_taste)


# For now, just keeping this in a single method, will likely refactor to a class since that makes more sense
def main():
	with open('eval.log', 'a') as logfile:
		logfile.write(f"Starting execution @ {datetime.datetime.now()}\n")

	# Initialize data/references to be used throughout evaluation
	songs = json.loads(open('data/clust_to_song_dict.json', 'r').readline())
	songs = {k: np.array(v) for k, v in songs.items()}
	data = synthetic.RecommendData()
	keys, features = synthetic.load_features('data/')
	retained_dict = keep_only_songs(keys, features, np.concatenate(list(songs.values()), axis=0))
	keys, features = np.array(list(retained_dict.keys())), np.array(list(retained_dict.values()))
	users = data.get_users(N_USERS, d=len(features[0]), small_world=True)
	clusters = [np.array(list(songlist)) for songlist in songs.values()]

	logger.info(f"Starting new recommendation evaluation with seed {SEED}.")

	logger.info(f"Basic recommendation with max_scores=0, max_ratings=0, min_similar=1 (forced always computing)")
	# Generate a basic recommendation set
	with model.RecommendDB(max_scores=0, max_ratings=0, min_similar=1, seed=SEED) as db:
		# Store preliminary data
		synthetic.store_features(db, keys, features)
		synthetic.store_users(db, *users)
		synthetic.store_clusters(db, *clusters)
		# Uncomment to run tsne plotting
		# plot_tsne(clusters, features, db)

		# Track timing for recommendation
		t, rec = timed_evaluation(db, rec_params=REC_CFG)

		# Can perform analysis on del_taste now if we find it is necessary.
		del_taste = test_n_biases(users[0], db, rec, 2)
		return 0

	# Time a few recommendations for fuzzy models
	for i in range(5):
		s = round(max(np.random.random(), 0.1), 3)

		logger.info(f"Fuzzy matching run #{i+1}, s={s}")
		db = model.RecommendDB(max_scores=0.1, max_ratings=0.1, min_similar=s)
		t, rec = timed_evaluation(db)

		logger.info(f"#{i+1} took {t}s\n")

if __name__ == '__main__':
	main()