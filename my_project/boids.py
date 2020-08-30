from manimlib.imports import *

class Boid(object):
	CONFIG = {
	}
	def __init__(self, position, velocity=None):
		self.position = position
		self.velocity = velocity or np.random.uniform(-1, 1, self.dimentions)

	def update(self, dt):
		self.shift(self.velocity * dt)
		self.velocity += self.acceleration * dt

class Boid2D(Boid, Dot):
	CONFIG = {
		"dimentions": 2,
	}

class Boid3D(Boid, Sphere):
	CONFIG = {
		"dimentions": 3,
	}

class Boids(object):
	def __init__(self, arg):
		# a bunch of boids
		# keep a list of them
		# handle the "lets find the neighbors for each of them"
		pass
