from manimlib.imports import *

class Boid(object):
	CONFIG = {
		# allowing a different radius for each force
		"radius_of_seperation": 1,
		"radius_of_alignment": 1,
		"radius_of_cohesion": 1,
		# TODO: use this angle for figuring out the fov
		"angle_of_seperation": PI,
		"angle_of_alignment": PI,
		"angle_of_cohesion": PI,

		# allowing different weights for each force
		"factor_seperation": 1,
		"factor_alignment": 1,
		"factor_cohesion": 1,


		"mass": 1, # resistance to forces : F=ma
	}
	def __init__(self, position, velocity, **kwargs):
		# digest_config(self, kwargs)

		self.position = position
		self.move_to(position)

		self.velocity = velocity
		# requires `set_rirection` method from Cone/Triangle
		self.set_direction(self.velocity)


	#
	# Boids main logic
	#
	def set_fov(self, fov):
		# expecting a list of boid objects
		self.fov = fov

	def get_force_seperation(self):
		"""
		calculate seperation based on self.fov
		return np.array with self.dimensions dimensions
		"""
		if self.radius_of_seperation == 0:
			return np.zeros(self.dimensions)
		# return np.array with self.dimensions dimensions
	def get_force_alignment(self):
		"""
		calculate alignment based on self.fov
		return np.array with self.dimensions dimensions
		"""
		if self.radius_of_alignment == 0:
			return np.zeros(self.dimensions)

		# currently using velocity as a vector
		# maybe I should use it only as a direction, and keep the norm of the vector constant
		return self.factor_alignment * np.mean([b.velocity for b in self.fov])
		# return np.array with self.dimensions dimensions
	def get_force_cohesion(self):
		"""
		calculate cohesion based on self.fov
		return np.array with self.dimensions dimensions
		"""
		if self.radius_of_cohesion == 0:
			return np.zeros(self.dimensions)

		# calculate Center Of Mass
		com = np.mean([b.position for b in self.fov])
		com_direction = com - self.position

		# lets create an attractive force (positive value)
		# which grows as the distance is large (proportional to r^x, where x>0)

		# F = factor * distance^1
		return self.factor_cohesion * com_direction**1

	def get_force(self):
		return self.get_force_cohesion() + self.get_force_alignment() + self.get_force_seperation()

	#
	# generic getter/setter
	#
	@property
	def acceleration(self):
		return self.get_force() / self.mass
	

	#
	# updating
	#
	def update(self, dt):
		self.update_position(dt)
		self.update_direction(dt)
		self.update_velocity(dt)
	def update_position(self, dt):
		self.shift(self.velocity * dt)
	def update_direction(self, dt):
		pass
	def update_velocity(self, dt):
		self.velocity += self.acceleration * dt




class Boid2D(Boid, Triangle):
	CONFIG = {
		"dimensions": 2,
	}
	def __init__(self, position, velocity, **kwargs):
		super().__init__(position, velocity, **kwargs)
		super(Boid, self).__init__(**kwargs)

	def set_direction(self, direction):
		pass

# TODO
# [ ] lower base radius
# 		its a good thing that there's no base
# [ ] lower the height
class Boid3D(Boid, Cone):
	CONFIG = {
		"dimensions": 3,
	}
	def __init__(self, position, velocity, **kwargs):
		# Cone constructor first
		super(Boid, self).__init__(**kwargs)
		# Then Boid constructor
		super().__init__(position, velocity, **kwargs)

class Boids(VMobject):
	CONFIG = {
		# limiting range as a cube to make the code more dimensions-generic
		# this is done in order to avoid x_min, y_min & z_min.
		"position_min": -3,
		"position_max":  3,

		"dimensions": 2, # either 2 or 3
	}
	def __init__(self, n=5, **kwargs):
		super().__init__(**kwargs)

		random_position = np.random.uniform(self.position_min, self.position_max, (n, self.dimensions))
		random_velocity = np.random.uniform(self.position_min, self.position_max, (n, self.dimensions))

		self.boids = [
			self._boid_class(random_position[i], random_velocity[i])
			for i in range(n)
		]
		
		self.add(*self.boids)

	@property
	def _boid_class(self):
		if self.dimensions == 2:
			return Boid2D
		elif self.dimensions == 3:
			return Boid3D
		else:
			raise ValueError("invalid dimensions value. Please enter either 2 or 3.")

	def update(self, dt):
		pass
		# for each boid, find its fov
		# then, for each boid, calculate force & propagate

class Boids2DScene(Scene):
	def construct(self):
		self.boids = Boids(dimensions=2)
		self.add(self.boids)
		self.wait()
		
class Boids3DScene(SpecialThreeDScene):
	def construct(self):
		import pdb; pdb.set_trace()
		self.boids = Boids(dimensions=3)
		self.add(self.boids)
		self.wait()
	