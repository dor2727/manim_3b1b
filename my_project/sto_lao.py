from manimlib.imports import *
from my_project.qubit_utils import *

OUTPUT_DIRECTORY = "STO_LAO"

class Perovskite(SpecialThreeDScene):
	CONFIG = {
		"three_d_axes_config": {
			"num_axis_pieces": 1,
			"number_line_config": {
				"unit_size": 2,
				"tick_frequency": 1,
				"numbers_with_elongated_ticks": [0, 1, 2],
				"stroke_width": 2,
			}
		},
		"init_camera_orientation": {
			"phi": 80 * DEGREES,
			# "theta": -135 * DEGREES,
			"theta": 15 * DEGREES,
		},
		"d": 1,
	}

	def construct(self):
		self.init_camera()
		self.init_axes()

		a1 = A()
		b = []
		for x in (X_AXIS*self.d/2, -X_AXIS*self.d/2):
			for y in (Y_AXIS*self.d/2, -Y_AXIS*self.d/2):
				for z in (Z_AXIS*self.d/2, -Z_AXIS*self.d/2):
					b.append(B(center=ORIGIN + x + y + z))
		x = []
		for d in (self.d/2, -self.d/2):
			for axis in (X_AXIS, Y_AXIS, Z_AXIS):
				x.append(X(center=ORIGIN + d*axis))

		self.add(a1, *b, *x)
		self.wait(1)

	def init_camera(self):
		self.set_camera_orientation(**self.init_camera_orientation)

	def init_axes(self):
		self.axes = self.get_axes()
		self.add(self.axes)

	def _tex(self, *s):
		tex = TexMobject(*s)
		tex.rotate(90 * DEGREES, RIGHT)
		tex.rotate(90 * DEGREES, OUT)
		tex.scale(0.5)
		return tex

	def rotate_to_surface(self, surface=[1,0,0]):
		if type(surface) is str:
			# parse
			pass

		vec = surface
		# plot vec as a line, not a vector, because of the tip, which is aligned with the xy plane
		# self.play( grow( vec ) )
		# rorate camera to show vec as a point in the vecter
		pass

class A(Sphere):
	CONFIG = {
        "radius": 0.12,
        "fill_color": BLUE_D,
        "fill_opacity": 1.0,
        "checkerboard_colors": [BLUE_D, BLUE_E],

        "center": ORIGIN,
	}
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.move_to(self.center)
class B(Sphere):
	CONFIG = {
        "radius": 0.12,
        "fill_color": GREEN_D,
        "checkerboard_colors": [GREEN_D, GREEN_E],

        "center": ORIGIN,
	}
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.move_to(self.center)
class X(Sphere):
	CONFIG = {
        "radius": 0.08,
        "fill_color": RED_D,
        "checkerboard_colors": [RED_D, RED_E],

        "center": ORIGIN,
	}
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.move_to(self.center)


"""
create a background grid
	similar to coordinate_systems.py
another useful file: three_d_sceene.py

vec = [1,0,0] or [1,1,0] or [1,1,1]
	then rotate as if we are looking straight at it
	to see the different surfaces (100) (110) (111)

create 3d vec - with a cone as its tip
"""
