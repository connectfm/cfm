import numpy as np
from scipy.spatial import distance
from sklearn import neighbors


class Simulation:
	__slots__ = [
		'n_users',
		'n_songs',
		'n_dims',
		'n_clusters',
		'r',
		'metric',
		'locations',
		'tastes',
		'songs',
		'ratings',
		'clusters',
		'biases',
		'knn',
		'_rng']

	def __init__(
			self,
			n_users=1_000,
			n_songs=1_000,
			n_dims=10,
			n_clusters=4,
			r=20,
			metric=distance.euclidean,
			seed=None):
		self.n_users = n_users
		self.n_songs = n_songs
		self.n_dims = n_dims
		self.n_clusters = n_clusters
		self.r = r
		self.metric = metric
		rng = np.random.default_rng
		self._rng = rng() if seed is None else rng(seed)
		self.locations = self._rng.lognormal(1, 2, (self.n_users, 2))
		self.tastes = self._rng.lognormal(1, 2, (self.n_users, self.n_dims))
		self.songs = self._rng.lognormal(1, 2, (self.n_songs, self.n_dims))
		self.ratings = self._rng.integers(1, 4, (self.n_users, self.n_songs))
		self.biases = self._rng.integers(0, 2, size=self.n_users)
		self.clusters = self._rng.integers(0, self.n_clusters, self.n_songs)
		kd_tree = neighbors.KDTree(self.locations)
		self.knn = kd_tree.query_radius(self.locations, self.r)

	def recommend(self):
		u = self._rng.integers(0, self.n_users)
		nn = self.knn[u]
		ne = self.sample_neighbor(u, nn)
		return u, self.sample_song(u, ne)

	def sample_neighbor(self, u, ne):
		me = np.array([self.locations[u]])
		ne = self.locations[ne]
		sims = 1 / (1 + distance.cdist(me, ne, metric=self.metric).flatten())
		norm = sum(sims)
		probs = sims / norm
		neighbor = self._rng.choice(ne, p=probs)
		np.argwhere(self.locations == neighbor)
		return neighbor

	def sample_song(self, u, v):
		clusters = []
		for c in range(self.n_clusters):
			cluster = np.argwhere(self.clusters == c).flatten()
			ratings = self.rating(u, v, cluster)
			clusters.append((c, sum(ratings)))
		norm = sum(r for _, r in clusters)
		probs = [c / norm for c, _ in clusters]
		cluster = self._rng.choice(clusters, p=probs)
		c, _ = cluster
		songs = np.argwhere(self.clusters == c).flatten()
		ratings = self.rating(u, v, songs)
		norm = sum(ratings)
		probs = ratings / norm
		song = self._rng.choice(songs, p=probs)
		return song

	def rating(self, u, v, s):
		b_u = self.biases[u]
		b_v = self.biases[v]
		num = (b_u * self.ratings[u, s]) + (b_v * self.ratings[v, s])
		den = (b_u * self.tastes[u]) + (b_v * self.tastes[v])
		return num / den


if __name__ == '__main__':
	sim = Simulation()
	sim.recommend()
