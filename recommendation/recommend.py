import datetime
import numbers
from typing import Any, Callable

import aioredis
import attr
import numpy as np
from scipy.spatial import distance

NOW = datetime.datetime.utcnow()
DAY_IN_SECS = 86_400

redis = await aioredis.create_redis_pool('redis://localhost')


@attr.s(slots=True)
class User:
	name = attr.ib(type=str)
	taste = attr.ib(type=np.ndarray, default=None)
	bias = attr.ib(type=numbers.Real, default=None)
	lat = attr.ib(type=numbers.Real, default=None)
	long = attr.ib(type=numbers.Real, default=None)
	rad = attr.ib(type=numbers.Real, default=None)
	rating = attr.ib(type=int, default=None)
	time = attr.ib(type=numbers.Real, default=None)

	def __attrs_post_init__(self):
		if self.taste is not None:
			self.taste = np.array(self.taste)

	def retrieve(self):
		attrs = [self.taste, self.bias, self.lat, self.long, self.rad]
		missing = np.array([a is None for a in attrs])
		missing = np.flatnonzero(missing)
		keys = np.array([
			f'{self.name}_taste',
			f'{self.name}_bias',
			f'{self.name}_lat',
			f'{self.name}_long',
			f'{self.name}_rad'])
		query = keys[missing]
		values = redis.mget(*query)
		for m, v in zip(missing, values):
			attrs[m] = v
		self.taste = np.array(self.taste)

	@classmethod
	def from_name(cls, name: str):
		user = User(name)
		user.retrieve()
		return user


@attr.s(slots=True, frozen=True)
class Recommender:
	"""Recommendation system for connect.fm"""
	metric = attr.ib(type=Callable, default=distance.euclidean)
	rng = attr.ib(type=np.random.Generator, default=np.random.default_rng())

	async def recommend(self, user: str) -> str:
		user = User.from_name(user)
		neighbors = await self.get_neighbors(user)
		if neighbors.size > 0:
			tastes = await self.get_tastes(*neighbors)
			neighbor = self.sample_neighbor(user, neighbors, tastes)
		else:
			neighbor = user
		song = await self.sample_song(user, neighbor)
		return song

	@staticmethod
	async def get_neighbors(user: User) -> np.ndarray:
		neighbors = redis.georadius(user.name, user.long, user.lat, user.rad)
		return np.array(neighbors)

	@staticmethod
	async def get_tastes(*users: str) -> np.ndarray:
		tastes = redis.mget(*(f'{user}_taste' for user in users))
		return np.array(tastes)

	def sample_neighbor(
			self,
			user: User,
			neighbors: np.ndarray,
			tastes: np.ndarray) -> User:
		u_taste = np.array([user.taste])
		dissimilarity = distance.cdist(u_taste, tastes, metric=self.metric)
		similarity = 1 / (1 + dissimilarity)
		neighbor, taste = self.sample(neighbors, similarity, with_weight=True)
		neighbor = User(neighbor, taste=taste)
		return neighbor

	def sample(
			self,
			population: np.ndarray,
			weights: np.ndarray,
			with_weight: bool = False) -> Any:
		probs = weights / sum(weights)
		if with_weight:
			population = np.vstack((np.arange(len(population)), population)).T
			idx_and_sample = self.rng.choice(population, p=probs)
			idx, sample = idx_and_sample[0], idx_and_sample[1:]
			sample = sample.item() if sample.shape == (1,) else sample
			sample = (sample, weights[idx])
		else:
			sample = self.rng.choice(population, p=probs)
		return sample

	async def sample_song(self, user: User, neighbor: User) -> str:
		cluster = await self.sample_cluster(user, neighbor)
		songs, ratings = [], []
		idx = 0
		async for song in redis.sscan(f'cluster_{cluster}'):
			rating = self.adj_rating(user, neighbor, song)
			songs[idx], ratings[idx] = song, rating
			idx += 1
		song = self.sample(np.array(songs), np.array(ratings))
		return song

	def adj_rating(self, user: User, neighbor: User, song: str) -> int:
		result = await redis.mget(
			song,
			f'{user.name}_{song}_rating',
			f'{neighbor.name}_{song}_rating',
			f'{user.name}_{song}_time',
			f'{neighbor.name}_{song}_time')
		feats, user.rating, neighbor.rating, user.time, neighbor.time = result
		dists = np.array([
			self.metric(user.taste, feats),
			self.metric(neighbor.taste, feats)])
		discounts = np.array([
			self.delta(user.time), self.delta(neighbor.time)])
		biases = np.array([user.bias, 1 - user.bias])
		ratings = np.array([user.rating, neighbor.rating])
		num = sum(biases * ratings * np.exp(-discounts))
		den = sum(biases * dists)
		rating = num / den
		return rating

	async def sample_cluster(self, user: User, neighbor: User) -> int:
		n_clusters = await redis.get('n_clusters')
		c_ratings = np.zeros(n_clusters)
		idx = 0
		async for cluster in redis.iscan('cluster_*'):
			async for song in redis.sscan(f'cluster_{cluster}'):
				rating = self.adj_rating(user, neighbor, song)
				c_ratings[idx] += rating
			idx += 1
		cluster = self.sample(np.arange(n_clusters), c_ratings)
		return cluster

	@staticmethod
	def delta(timestamp: float) -> float:
		delta = NOW - datetime.datetime.utcfromtimestamp(timestamp)
		delta = delta.total_seconds() / DAY_IN_SECS
		return delta
