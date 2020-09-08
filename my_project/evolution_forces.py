from manimlib.imports import *

class ParaboloidPlot(SpecialThreeDScene):
	CONFIG = {
		"three_d_axes_config": {
			"num_axis_pieces": 1,
			"number_line_config": {
				"unit_size": 2,
				"tick_frequency": 1,
				"numbers_with_elongated_ticks": [0, 1, 2],
				"stroke_width": 2,
			},
			"axis_config": {
				"unit_size": 1,
				"tick_frequency": 1,
				"numbers_with_elongated_ticks": [],
				"stroke_width": 2,
			},
			"x_min": 0,
			"x_max": 7,
			"y_min": 0,
			"y_max": 7,
			"z_min": 0,
			"z_max": 7,
		},
		"init_camera_orientation": {
			"phi": 80 * DEGREES,
			# "theta": -135 * DEGREES,
			"theta": 290 * DEGREES,
		},
		"paraboloid_config": {
			"r_max":  1,
			"center_point": 2*X_AXIS + 2*Y_AXIS,
		},
		"axes_center_point": -2.5*X_AXIS - 2.5*Y_AXIS - 0.75*Z_AXIS,
	}

	def construct(self):
		self.init_camera()
		self.init_axes()

		paraboloid = ParaboloidPolar(**self.paraboloid_config)
		paraboloid.shift(self.axes_center_point)
		self.add(paraboloid)

		x, y = 2.1, 2.9
		z = paraboloid.get_value_at_point([x,y])
		point = np.array([x,y,z])
		sphere = Sphere(radius=0.05, fill_color=WHITE, checkerboard_colors=False)
		sphere.shift(point)
		sphere.shift(self.axes_center_point)
		self.add(sphere)

		# going 20 degrees in 2 seconds
		# 60 frames per seconds
		# 20 degrees in 120 frames
		rate = 20/120
		# it won't reach exactly 60 degrees, but it'll be close enough
		self.begin_ambient_camera_rotation(rate=-rate, about="phi")
		self.wait(2)
		self.stop_ambient_camera_rotation(about="phi")

		direction = - paraboloid.get_slope_at_point(point)
		print(point)
		print(direction)
		print(point + direction)
		force = Arrow3d(start=point, end=point + direction)
		force.shift(self.axes_center_point)
		self.add(force)

		self.wait()

	def init_camera(self):
		self.set_camera_orientation(**self.init_camera_orientation)

	def init_axes(self):
		self.axes = self.get_axes()
		self.axes.x_axis.set_color(BLUE)
		self.axes.y_axis.set_color(GREEN)
		self.axes.z_axis.set_color(RED)
		# self.set_axes_labels()
		self.axes.shift(self.axes_center_point)
		self.add(self.axes)
