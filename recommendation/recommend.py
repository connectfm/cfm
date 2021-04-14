from collections import Callable
from typing import Any, Tuple

import aioredis
import attr
import numpy as np
from scipy.spatial import distance

from recommendation import synthetic

redis = await aioredis.create_redis_pool('redis://localhost')
Taste = np.ndarray
User = str
Neighbors = np.ndarray
Song = str
Rating = int
Bias = float


@attr.s(slots=True, frozen=True)
class Recommender:
	metric = attr.ib(type=Callable, default=distance.euclidean)
	rng = attr.ib(type=np.random.Generator, default=np.random.default_rng)

	async def recommend(self, u: User):
		ne = await self.get_neighbors(u)
		u_taste, ne_tastes = await self.get_tastes(u, *ne)
		if ne.size:
			ne, ne_taste = self.sample_neighbor(ne, u_taste, ne_tastes)
		else:
			ne, ne_taste = u, u_taste
		song = await self.sample_song(u, ne, u_taste, ne_taste)
		return song

	async def get_neighbors(self, u: User) -> Neighbors:
		long, lat, r = await redis.mget(*(f'{u}_long', f'{u}_lat', f'{u}_r'))
		ne = redis.georadius(u, long, lat, r)
		return ne

	async def get_tastes(self, u: User, *ne: User) -> Tuple[Taste, Taste]:
		tastes = redis.mget(*(f'{n}_taste' for n in zip(u, *ne)))
		u_taste, ne_tastes = np.array(tastes[0]), np.array(tastes[1:])
		return u_taste, ne_tastes

	def sample_neighbor(
			self,
			ne: Neighbors,
			u_taste: Taste,
			ne_tastes: Taste) -> Tuple[User, Taste]:
		u_taste = np.array([u_taste])
		diff = distance.cdist(u_taste, ne_tastes, metric=self.metric)
		sim = 1 / (1 + diff)
		ne, taste = self.sample(ne, sim, with_weight=True)
		return ne, taste

	async def sample_song(
			self, u: User, ne: User, u_taste: Taste, ne_taste: Taste) -> Song:
		bias = redis.get(f'{u}_bias')
		cluster = self.sample_cluster(u, ne, u_taste, ne_taste, bias)
		songs, ratings = [], []
		idx = 0
		async for s in redis.sscan(f'cluster_{cluster}'):
			u_rating, ne_rating, s_features = redis.mget(
				f'{u}_{s}_rating', f'{ne}_{s}_rating', f'{s}_features')
			rating = self.adjusted_rating(
				s_features, u_taste, ne_taste, u_rating, ne_rating, bias)
			songs[idx] = s
			ratings[idx] = rating
			idx += 1
		song = self.sample(np.array(songs), np.array(ratings))
		return song

	async def sample_cluster(
			self,
			u: User,
			ne: User,
			u_taste: Taste,
			ne_taste: Taste,
			bias: Bias):
		n_clusters = redis.get('n_clusters')
		c_ratings = np.zeros(n_clusters)
		async for cluster in redis.iscan('cluster_*'):
			c = int(cluster.split()[-1])
			async for s in redis.sscan(cluster):
				u_rating, ne_rating, s_features = redis.mget(
					f'{u}_{s}_rating', f'{ne}_{s}_rating', s)
				rating = self.adjusted_rating(
					s_features, u_taste, ne_taste, u_rating, ne_rating, bias)
				c_ratings[c] += rating
		cluster = self.sample(np.arange(n_clusters), c_ratings)
		return cluster

	def adjusted_rating(
			self,
			song: Taste,
			u_taste: Taste,
			ne_taste: Taste,
			u_rating: Rating,
			ne_rating: Rating,
			bias: Bias):
		u_dist = self.metric(u_taste, song)
		ne_dist = self.metric(ne_taste, song)
		num = (bias * u_rating) + ((1 - bias) * ne_rating)
		den = (bias * u_dist) + ((1 - bias) * ne_dist)
		rating = num / den
		return rating

	def sample(
			self,
			pop: np.ndarray,
			weights: np.ndarray,
			with_weight: bool = False) -> Any:
		probs = weights / sum(weights)
		if with_weight:
			pop = np.vstack(np.arange(len(pop)), pop).T
			idx_and_sample = self.rng.choice(pop, p=probs)
			idx, sample = idx_and_sample[0], idx_and_sample[1:]
			sample = (sample, weights[idx])
		else:
			sample = self.rng.choice(pop, p=probs)
		return sample


if __name__ == '__main__':
	u_rating = 1
	ne_rating = 3
	u_taste = synthetic.get_taste()
	ne_taste = synthetic.get_taste()
	song = synthetic.get_taste()
	rec = Recommender()
	rating = rec.adjusted_rating(
		song=song,
		u_taste=u_taste,
		ne_taste=ne_taste,
		u_rating=u_rating,
		ne_rating=ne_rating,
		bias=0.5)
	print(rating)
