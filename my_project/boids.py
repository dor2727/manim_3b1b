from manimlib.imports import *
import functools

class Boid(object):
	CONFIG = {
		# allowing a different radius for each force
		"radius_of_seperation": 1,
		"radius_of_alignment": 1,
		"radius_of_cohesion": 1,
		# TODO: use this angle for figuring out the fov
		"angle_of_seperation": PI, # currently not used
		"angle_of_alignment": PI, # currently not used
		"angle_of_cohesion": PI, # currently not used

		# allowing different weights for each force
		"factor_seperation": 1,
		"factor_seperation_object": 1,
		"factor_alignment": 1,
		"factor_cohesion": 1,

		# allowing different powers for each force
		"seperation_object_power": -1,
		"seperation_power": -1,
		"alignment_power": 1,
		"cohesion_power": 1,

		"speed_mean": 1,
		"speed_std": 0, # if set to 0, then the speed will stay constant

		"mass": 1, # resistance to forces : F=ma
	}
	def __init__(self, position, velocity, **kwargs):
		# there's no need to call digest config, since this class is a meta class
		# Thus, Boid2D / Boid3D will inherit from a Mobject, and will call digest_config on their own

		self.move_to(position)

		self.set_speed(kwargs.get("speed", self.speed_mean))
		self.set_velocity(velocity)

		# requires `set_direction` method from Cone/Triangle/Boid2D
		self.set_direction(self.velocity)

	#
	# Getting nearby boids
	#
	def set_other_boids(self, boids):
		if self in boids:
			# create a copy of the list, so that we won't change it
			boids = boids[::]
			boids.remove(self)

		self._other_boids = boids
	def get_other_boids(self):
		return getattr(self, "_other_boids", [])
	#
	# Getting nearby objects
	#
	def set_other_objects(self, objects):
		self._other_objects = objects
	def get_other_objects(self):
		return getattr(self, "_other_objects", [])

	#
	# Field Of View & Distance utils
	#
	def is_boid_in_fov(self, other_boid, radius, angle):
		if radius == 0:
			return False

		distance = self.get_boid_distance(other_boid)
		if get_norm(distance) < radius:
			if angle == PI: # full circle
				return True

			# TODO: test this code
			# following https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors
			vec_1 = distance
			vec_2 = self.velocity
			dot = vec_1.dot(vec_2)
			size_1 = get_norm(vec_1)
			size_2 = get_norm(vec_2)
			other_boid_angle = np.arccos(dot / (size_1 * size_2))
			return other_boid_angle <= angle

		return False
	def get_boid_distance(self, b):
		return b.get_center() - self.get_center()

	def is_object_in_fov(self, obj):
		distance = self.get_object_distance(obj)
		return get_norm(distance) <= self.radius_of_seperation
	def get_object_distance(self, obj):
		if type(obj) is LineX:
			distance = Y_AXIS * (obj.y - self.get_center()[1])

			# check if self is in the X range of the line
			x_min = min(obj.x_start, obj.x_end)
			x_max = max(obj.x_start, obj.x_end)

			if self.get_center()[0] > x_max:
				distance += X_AXIS * (self.get_center()[0] - x_max)
			elif self.get_center()[0] < x_min:
				distance += X_AXIS * (x_max - self.get_center()[0])

		elif type(obj) is LineY:
			distance = X_AXIS * (obj.x - self.get_center()[0])

			# check if self is in the X range of the line
			y_min = min(obj.y_start, obj.y_end)
			y_max = max(obj.y_start, obj.y_end)

			if self.get_center()[1] > y_max:
				distance += Y_AXIS * (self.get_center()[1] - y_max)
			elif self.get_center()[1] < y_min:
				distance += Y_AXIS * (y_min - self.get_center()[1])
		elif type(obj) is Circle:
			raise NotImplemented
			# Circle has get_center() & radius
			# size of distance is get_norm(self.get_center() - Circle.get_center()) - radius
			# direction is self.get_center() - Circle.get_center()
			# not sure about that
			distance = np.array([np.inf]*3)
		else:
			distance = np.array([np.inf]*3)

		return distance

	#
	# Getting Field Of View
	#
	def set_fov(self):
		self.fov_seperation = list(filter(
			lambda b: self.is_boid_in_fov(b, self.radius_of_seperation, self.angle_of_seperation),
			self.get_other_boids()
		))
		self.fov_alignment  = list(filter(
			lambda b: self.is_boid_in_fov(b, self.radius_of_alignment , self.angle_of_alignment ),
			self.get_other_boids()
		))
		self.fov_cohesion   = list(filter(
			lambda b: self.is_boid_in_fov(b, self.radius_of_cohesion  , self.angle_of_cohesion  ),
			self.get_other_boids()
		))

		self.fov_seperation_objects = list(filter(
			self.is_object_in_fov,
			self.get_other_objects()
		))
	def get_fov_seperation(self):
		return getattr(self, "fov_seperation", [])
	def get_fov_alignment(self):
		return getattr(self, "fov_alignment", [])
	def get_fov_cohesion(self):
		return getattr(self, "fov_cohesion", [])
	def get_fov_seperation_objects(self):
		return getattr(self, "fov_seperation_objects", [])

	#
	# calculating the different forces
	#
	def calculate_force(self, factor, vector, power, repel=False):
		sign = (-1) ** repel
		direction = sign * normalize(vector)
		strength = factor * get_norm(vector)**power
		return strength * direction
	def get_force_seperation_boids(self):
		return sum(
			(self.calculate_force(
				factor = self.factor_seperation,
				vector = self.get_boid_distance(b),
				power  = self.seperation_power,
				repel  = True,
			) for b in self.get_fov_seperation()),
			np.zeros(3)
		)
	def get_force_seperation_objects(self):
		return sum(
			(self.calculate_force(
				factor = self.factor_seperation,
				vector = self.get_object_distance(obj),
				power  = self.seperation_object_power,
				repel  = True,
			) for obj in self.get_fov_seperation_objects()),
			np.zeros(3)
		)
	def get_force_alignment(self):
		if self.radius_of_alignment == 0 or not self.get_fov_alignment():
			return np.zeros(3)

		velocity_direction = np.mean([b.velocity for b in self.get_fov_alignment()], axis=0)

		return self.calculate_force(self.factor_alignment, velocity_direction, self.alignment_power, repel=False)
	def get_force_cohesion(self):
		if self.radius_of_cohesion == 0 or not self.get_fov_cohesion():
			return np.zeros(3)

		# calculate Center Of Mass
		com_direction = np.mean([self.get_boid_distance(b) for b in self.get_fov_cohesion()], axis=0)

		return self.calculate_force(self.factor_cohesion, com_direction, self.cohesion_power, repel=False)

	def get_force_seperation(self):
		return self.get_force_seperation_boids() + self.get_force_seperation_objects()
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
	def update_velocity(self, dt):
		# this will be the new velocity direction
		increased_velocity = self.velocity + self.acceleration * dt
		# however, we allow for a specific range of speeds
		self.set_speed(get_norm(increased_velocity), normalize=True)
		self.set_velocity(increased_velocity)
	def update_direction(self):
		self.set_direction(self.velocity)

	#
	# Setting Velocity (vector) and Speed (magnitude)
	#
	def set_velocity(self, velocity):
		self.velocity = normalize(velocity) * self.speed
	def set_speed(self, new_speed, normalize=False):
		if normalize:
			# we take the difference between the new speed & the current one
			diff = new_speed - self.speed
			# Then we normalize it to a number between -1 to 1, using the sigmoid function
			# 	(for no particular reason. It simply has a nice behavior)
			normalized_diff = 2 * ((np.exp(diff) / (np.exp(diff) + 1)) - 0.5)
			# Next, convert normalized_diff to a percentage, based on the distance from the min/max allowed speed
			if normalized_diff < 0:
				allowed_change = self.speed - self.min_allowed_speed
			elif normalized_diff > 0:
				allowed_change = self.max_allowed_speed - self.speed
			else:
				allowed_change = 0
			# the new speed is the current one plus or minus (depending on the sign of normalized_diff)
			# some fraction of the allowed changed (the fraction is expressed in normalized diff)
			self.speed += normalized_diff * allowed_change
		else:
			self.speed = new_speed

	#
	# Speed getter, setter, and other utils
	#
	@property
	def speed(self):
		return getattr(self, "_speed", self.speed_mean)
	@speed.setter
	def speed(self, value):
		# verifying speed range
		if value < self.min_allowed_speed:
			print("[!] a speed lower than the minimum:", value)
			value = self.min_allowed_speed
		if value > self.max_allowed_speed:
			print("[!] a speed higher than the maximum:", value)
			value = self.max_allowed_speed

		self._speed = value

		## Setting a color to represent the velocity
		# normalize to a number between 0 and 1
		normalized_speed = (value - self.speed_mean + self.speed_std) / (2 * self.speed_std)
		# then use it to get one of the colors
		self.set_color(self.speed_color_gradient[int(normalized_speed * 20)])
	@property
	def min_allowed_speed(self):
		return self.speed_mean - self.speed_std
	@property
	def max_allowed_speed(self):
		return self.speed_mean + self.speed_std
	@property
	@functools.lru_cache()
	def speed_color_gradient(self):
		return color_gradient([BLUE_A, BLUE_E, RED_E, RED_A], 20)




class Boid2D(Boid, ArrowTip): # ArrowTip is Triangle++
	CONFIG = {
		"dimensions": 2,

		"triangle_config": {},
	}
	def __init__(self, position, velocity, **kwargs):
		# parse CONFIG
		digest_config(self, kwargs)
		# ArrowTip constructor first
		super(Boid, self).__init__(**self.triangle_config, **kwargs)
		# Then Boid constructor
		super().__init__(position, velocity, **kwargs)

	# in the 2d case, `direction` is identical to `angle`. Yet, `direction` is used for generality and compatability with 3D
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
		"x_min": -7,
		"x_max":  7,
		"y_min": -3.5,
		"y_max":  3.5,
		"z_min": -3,
		"z_max":  3,

		"boid_config": {
			"speed_mean": 1.5,
			"speed_std": 0.5,
		},

		"dimensions": 2, # either 2 or 3
	}
	def __init__(self, n=5, **kwargs):
		super().__init__(**kwargs)
		self.init_boids(n)

		# for b in self.boids:
		# 	b.add_updater(self.debug)
		# 	b.add_updater(self.stay_in_the_box)
		# self.for_each(Boid.add_updater, self.debug)
		# self.for_each(Boid.add_updater, self.stay_in_the_box)

	@property
	@functools.lru_cache()
	def _boid_class(self):
		if self.dimensions == 2:
			return Boid2D
		elif self.dimensions == 3:
			return Boid3D
		else:
			raise ValueError("invalid dimensions value. Please enter either 2 or 3.")

	def _generate_random_positions(self, n):
		# each vector has to be 3d
		random_position = np.array([
			np.random.uniform(self.x_min, self.x_max, n),
			np.random.uniform(self.y_min, self.y_max, n),
			np.random.uniform(self.z_min, self.z_max, n),
		]).T

		# then, we truncate the unwanted dimensions
		random_position[:,self.dimensions:] = np.zeros((n,3-self.dimensions))

		return random_position
	def _generate_random_velocities(self, n):
		# first, choose direction
		random_velocity = np.random.uniform(-1, 1, (n, 3))
		random_velocity[:,self.dimensions:] = np.zeros((n,3-self.dimensions))

		# Then, generate random velocity size (the length of the vector, i.e. the speed)
		mean = self.boid_config["speed_mean"]
		std  = self.boid_config["speed_std"]
		random_speed = np.random.uniform(mean - std, mean + std, n)

		return random_velocity, random_speed

	def init_boids(self, n):
		random_position = self._generate_random_positions(n)
		random_velocity, random_speed = self._generate_random_velocities(n)
		
		self.boids = []

		for i in range(n):
			b = self._boid_class(
				random_position[i],
				random_velocity[i],
				speed=random_speed[i],
				**self.boid_config,
			)
			b.add_updater(b.__class__.updater)
			self.boids.append(b)
		
		self.for_each(Boid.set_other_boids, self.boids)

		self.add(*self.boids)

	# updaters
	def debug(self, boid_self, dt):
		print(boid_self.get_center(), boid_self.velocity)

	def stay_in_the_box(self, boid_self, dt):
		x_size = self.x_max - self.x_min
		y_size = self.y_max - self.y_min
		z_size = self.z_max - self.z_min
		# very manual check
		if boid_self.get_center()[0] > self.x_max:
			boid_self.shift(-x_size*X_AXIS)
		if boid_self.get_center()[0] < self.x_min:
			boid_self.shift( x_size*X_AXIS)

		if boid_self.get_center()[1] > self.y_max:
			boid_self.shift(-y_size*Y_AXIS)
		if boid_self.get_center()[1] < self.y_min:
			boid_self.shift( y_size*Y_AXIS)

		if boid_self.get_center()[2] > self.z_max:
			boid_self.shift(-z_size*Z_AXIS)
		if boid_self.get_center()[2] < self.z_min:
			boid_self.shift( z_size*Z_AXIS)
		return

		for b in self.boids:
			# very manual check
			if b.get_center()[0] > self.x_max:
				b.shift(-box_size*X_AXIS)
			if b.get_center()[0] < self.x_min:
				b.shift( box_size*X_AXIS)

			if b.get_center()[1] > self.y_max:
				b.shift(-box_size*Y_AXIS)
			if b.get_center()[1] < self.y_min:
				b.shift( box_size*Y_AXIS)

			if b.get_center()[2] > self.z_max:
				b.shift(-box_size*Z_AXIS)
			if b.get_center()[2] < self.z_min:
				b.shift( box_size*Z_AXIS)

	def for_each(self, method, *args, **kwargs):
		# the lazy version of a for loop
		for b in self.boids:
			method(b, *args, **kwargs)


###################
#### Obstacles ####
###################

# a line parallel to the x axis, i.e. y doesn't change
class LineX(Line):
	def __init__(self, y, x_start, x_end, **kwargs):
		start = x_start*X_AXIS + y*Y_AXIS
		end   = x_end  *X_AXIS + y*Y_AXIS
		super().__init__(start, end, **kwargs)
		self.y       = y
		self.x_start = x_start
		self.x_end   = x_end
# a line parallel to the y axis, i.e. x doesn't change
class LineY(Line):
	def __init__(self, x, y_start, y_end, **kwargs):
		start = x*X_AXIS + y_start*Y_AXIS
		end   = x*X_AXIS + y_end  *Y_AXIS
		super().__init__(start, end, **kwargs)
		self.x       = x
		self.y_start = y_start
		self.y_end   = y_end


################
#### Scenes ####
################

class Boids2DScene(Scene):
	CONFIG = {
		"objstacles": [
			LineY(3, -1, 1),
			LineX(1, -5, -4),
		],

		"boids_config": {
			"boid_config": {},
		},

		"n": 15, # amount of boids
		"duration": 15, # animation duration in seconds
	}
	def construct(self):
		# create boids
		self.boids = Boids(n=self.n, dimensions=2, **self.boids_config)
		self.add(self.boids)

		# create objects / obstacles
		self.add_boundary()
		self.add(*self.objstacles)

		# inform the boids about the objects
		self.boids.for_each(Boid.set_other_objects, self.borders + self.objstacles)

		# add axes for debug purpose
		self.add_axes()

		self.wait(self.duration)

	def add_boundary(self):
		l1 = LineY(self.boids.x_min, self.boids.y_min, self.boids.y_max)
		l2 = LineY(self.boids.x_max, self.boids.y_min, self.boids.y_max)
		l3 = LineX(self.boids.y_min, self.boids.x_min, self.boids.x_max)
		l4 = LineX(self.boids.y_max, self.boids.x_min, self.boids.x_max)
		self.borders = [l1, l2, l3, l4]
		self.add(*self.borders)

	def add_axes(self):
		axes = self.axes = Axes()
		# axes.set_stroke(width=2)
		axes.add_coordinates()

		self.add(axes)

# 3D is still a Work In Progress
class Boids3DScene(SpecialThreeDScene):
	CONFIG = {
		"objstacles": [],

		"n": 5, # amount of boids
		"duration": 5, # animation duration in seconds
	}
	def construct(self):
		self.boids = Boids(n=self.n, dimensions=3)
		self.add(self.boids)

		# inform the boids about the objects
		# self.boids.for_each(Boid.set_other_objects, self.borders + self.objstacles)

		self.wait(self.duration)

	def add_boundary(self):
		p_xy_top    = PlaneXY(self.boids.z_max, self.boids.x_min, self.boids.x_max, self.boids.y_min, self.boids.y_max)
		p_xy_bottom = PlaneXY(self.boids.z_min, self.boids.x_min, self.boids.x_max, self.boids.y_min, self.boids.y_max)
		p_xz_top    = PlaneXZ(self.boids.y_max, self.boids.x_min, self.boids.x_max, self.boids.z_min, self.boids.z_max)
		p_xz_bottom = PlaneXZ(self.boids.y_min, self.boids.x_min, self.boids.x_max, self.boids.z_min, self.boids.z_max)
		p_yz_top    = PlaneYZ(self.boids.x_max, self.boids.y_min, self.boids.y_max, self.boids.z_min, self.boids.z_max)
		p_yz_bottom = PlaneYZ(self.boids.x_min, self.boids.y_min, self.boids.y_max, self.boids.z_min, self.boids.z_max)
		self.borders = [p_xy_top, p_xy_bottom, p_xz_top, p_xz_bottom, p_yz_top, p_yz_bottom]
		self.add(*self.borders)


class Boids2DScene_1(Boids2DScene):
	CONFIG = {
		"boids_config": {
			"boid_config": {
				# allowing a different radius for each force
				"radius_of_seperation": 1,
				"radius_of_alignment": 2,
				"radius_of_cohesion": 2.5,
				# TODO: use this angle for figuring out the fov
				"angle_of_seperation": PI, # currently not used
				"angle_of_alignment": PI, # currently not used
				"angle_of_cohesion": PI, # currently not used

				# allowing different weights for each force
				"factor_seperation": 2.5,
				"factor_seperation_object": 1,
				"factor_alignment": 1.5,
				"factor_cohesion": 1.5,

				# allowing different powers for each force
				"seperation_object_power": -1,
				"seperation_power": -1,
				"alignment_power": 1,
				"cohesion_power": 1.3,

				"speed_mean": 1,
				"speed_std": 0.5, # if set to 0, then the speed will stay constant
			},
		},
		"n": 15,
		"duration": 60,
	}
class Boids2DScene_2(Boids2DScene):
	CONFIG = {
		"boids_config": {
			"boid_config": {
				# allowing a different radius for each force
				"radius_of_seperation": 1,
				"radius_of_alignment": 2,
				"radius_of_cohesion": 2.5,
				# TODO: use this angle for figuring out the fov
				"angle_of_seperation": PI, # currently not used
				"angle_of_alignment": PI, # currently not used
				"angle_of_cohesion": PI, # currently not used

				# allowing different weights for each force
				"factor_seperation": 1,
				"factor_seperation_object": 1,
				"factor_alignment": 1.5,
				"factor_cohesion": 1.5,

				# allowing different powers for each force
				"seperation_object_power": -1,
				"seperation_power": -1.25,
				"alignment_power": 1,
				"cohesion_power": 1.5,

				"speed_mean": 1,
				"speed_std": 0.5, # if set to 0, then the speed will stay constant
			},
		},
		"n": 15,
		"duration": 60,
	}
