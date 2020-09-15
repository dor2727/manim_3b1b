from manimlib.imports import *
import functools

class Boid(object):
	CONFIG = {
		# allowing a different radius for each force
		"radius_of_seperation": 1,
		"radius_of_alignment": 1, # currently not used
		"radius_of_cohesion": 1, # currently not used
		# TODO: use this angle for figuring out the fov
		"angle_of_seperation": PI, # currently not used
		"angle_of_alignment": PI, # currently not used
		"angle_of_cohesion": PI, # currently not used

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

		# requires `set_direction` method from Cone/Triangle
		self.set_direction(self.velocity)

	#
	# Getting nearby boids
	#
	def set_other_boids(self, boids):
		self._other_boids = boids
	def get_other_boids(self):
		return getattr(self, "_other_boids", [])

	def set_fov(self):
		fov = []

		for b in self.get_other_boids():
			if b is self:
				continue

			r2 = np.square(b.position - self.position).sum()
			if r2 < self.radius_of_seperation**2:
				# TODO: check theta
				if True:
					fov.append(b)

		self.fov = fov
	def get_fov(self):
		return getattr(self, "fov", [])

	#
	# calculating the different forces
	#
	def get_force_seperation(self):
		"""
		calculate seperation based on self.fov
		return np.array with self.dimensions dimensions
		"""
		if self.radius_of_seperation == 0 or not self.get_fov():
			return np.zeros(3)

		# place holder
		return np.zeros(3)
	def get_force_alignment(self):
		"""
		calculate alignment based on self.fov
		return np.array with self.dimensions dimensions
		"""
		if self.radius_of_alignment == 0 or not self.get_fov():
			return np.zeros(3)

		# currently using velocity as a vector
		# maybe I should use it only as a direction, and keep the norm of the vector constant
		return self.factor_alignment * np.mean([b.velocity for b in self.fov])
	def get_force_cohesion(self):
		"""
		calculate cohesion based on self.fov
		return np.array with self.dimensions dimensions
		"""
		if self.radius_of_cohesion == 0 or not self.get_fov():
			return np.zeros(3)

		# calculate Center Of Mass
		com = np.mean([b.position for b in self.fov])
		com_direction = com - self.position

		# lets create an attractive force (positive value)
		# which grows as the distance is large (proportional to r^x, where x>0)

		# F = factor * distance^1
		return self.factor_cohesion * com_direction**1


	def get_force(self):
		return self.get_force_cohesion() + self.get_force_alignment() + self.get_force_seperation()
	@property
	def acceleration(self):
		return self.get_force() / self.mass
	

	#
	# updating
	#
	def updater(self, dt):
		# update fov
		self.set_fov()
		# move according to the old velocity
		self.update_position(dt)
		# update the velocity according to the current acceleration
		self.update_velocity(dt)
		# re-align to the current (updated) velocity
		self.update_direction()
	def update_position(self, dt):
		self.shift(self.velocity * dt)
		self.position += self.velocity * dt
	def update_velocity(self, dt):
		self.velocity += self.acceleration * dt
	def update_direction(self):
		self.set_direction(self.velocity)




class Boid2D(Boid, ArrowTip): # ArrowTip is Triangle++
	CONFIG = {
		"dimensions": 2,

		"triangle_config": {
			"base_radius": 0.2,
			"height": 0.5,
		},
		"boid_config": {},
	}
	def __init__(self, position, velocity, **kwargs):
		# parse CONFIG
		digest_config(self, kwargs)
		# ArrowTip constructor first
		super(Boid, self).__init__(**self.triangle_config, **kwargs)
		# Then Boid constructor
		super().__init__(position, velocity, **self.boid_config, **kwargs)

	def set_direction(self, direction):
		new_angle = angle_of_vector(direction)
		old_angle = self.get_angle()
		self.rotate(new_angle - old_angle)
	def get_direction(self):
		return self.get_angle()
	

class Boid3D(Boid, Cone):
	CONFIG = {
		"dimensions": 3,

		"cone_config": {
			"base_radius": 0.2,
			"height": 0.5,
		},
		"boid_config": {},
		
	}
	def __init__(self, position, velocity, **kwargs):
		# parse CONFIG
		digest_config(self, kwargs)
		# Cone constructor first
		super(Boid, self).__init__(**self.cone_config, **kwargs)
		# Then Boid constructor
		super().__init__(position, velocity, **self.boid_config, **kwargs)

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
		self.init_boids(n)
		for b in self.boids:
			b.add_updater(self.stay_in_the_box)

	@property
	@functools.lru_cache()
	def _boid_class(self):
		if self.dimensions == 2:
			return Boid2D
		elif self.dimensions == 3:
			return Boid3D
		else:
			raise ValueError("invalid dimensions value. Please enter either 2 or 3.")

	def init_boids(self, n):
		# each vector has to be 3d
		random_position = np.random.uniform(self.position_min, self.position_max, (n, 3))
		random_velocity = np.random.uniform(self.position_min, self.position_max, (n, 3))
		# then, we truncate the unwanted dimensions
		random_position[:,self.dimensions:] = np.zeros((n,3-self.dimensions))
		random_velocity[:,self.dimensions:] = np.zeros((n,3-self.dimensions))

		self.boids = []

		for i in range(n):
			b = self._boid_class(random_position[i], random_velocity[i])
			b.add_updater(b.__class__.updater)
			# b.suspend_updating()
			self.boids.append(b)
		
		for b in self.boids:
			b.set_other_boids(self.boids)

		self.add(*self.boids)

	def stay_in_the_box(self, boid_self, dt):
		box_size = self.position_max - self.position_min
		# very manual check
		if boid_self.position[0] > self.position_max:
			boid_self.shift(-box_size*X_AXIS)
			boid_self.position += -box_size*X_AXIS
		if boid_self.position[0] < self.position_min:
			boid_self.shift( box_size*X_AXIS)
			boid_self.position +=  box_size*X_AXIS

		if boid_self.position[1] > self.position_max:
			boid_self.shift(-box_size*Y_AXIS)
			boid_self.position += -box_size*Y_AXIS
		if boid_self.position[1] < self.position_min:
			boid_self.shift( box_size*Y_AXIS)
			boid_self.position +=  box_size*Y_AXIS

		if boid_self.position[2] > self.position_max:
			boid_self.shift(-box_size*Z_AXIS)
			boid_self.position += -box_size*Z_AXIS
		if boid_self.position[2] < self.position_min:
			boid_self.shift( box_size*Z_AXIS)
			boid_self.position +=  box_size*Z_AXIS
		return

		for b in self.boids:
			# very manual check
			if b.position[0] > self.position_max:
				b.shift(-box_size*X_AXIS)
				b.position += -box_size*X_AXIS
			if b.position[0] < self.position_min:
				b.shift( box_size*X_AXIS)
				b.position +=  box_size*X_AXIS

			if b.position[1] > self.position_max:
				b.shift(-box_size*Y_AXIS)
				b.position += -box_size*Y_AXIS
			if b.position[1] < self.position_min:
				b.shift( box_size*Y_AXIS)
				b.position +=  box_size*Y_AXIS

			if b.position[2] > self.position_max:
				b.shift(-box_size*Z_AXIS)
				b.position += -box_size*Z_AXIS
			if b.position[2] < self.position_min:
				b.shift( box_size*Z_AXIS)
				b.position +=  box_size*Z_AXIS

			

class Boids2DScene(Scene):
	def construct(self):
		self.boids = Boids(n=5, dimensions=2)
		self.add(self.boids)
		self.boids.resume_updating()
		self.wait(40)
		
class Boids3DScene(SpecialThreeDScene):
	def construct(self):
		self.boids = Boids(n=5, dimensions=3)
		self.add(self.boids)
		# self.boids.resume_updating()
		self.wait(40)
	