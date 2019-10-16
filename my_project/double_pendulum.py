from manimlib.imports import *

OUTPUT_DIRECTORY = "double_pendulum"

class Example(Scene):
	CONFIG = {
		"pendulum_config": {
			"top_point": 2*UP,
			"L1": 2,
			"L2": 2,
			"m1": 1,
			"m2": 1,
			"initial_theta1": 45*DEGREES,
			"initial_theta2": 90*DEGREES,
			"initial_omega1": 0,
			"initial_omega2": 0,
			"gravity": 9.8,
			"weight_diameter": 0.5,
			"n_steps_per_frame": 100,
		},
		"run_time": 10,
	}
	def construct(self):
		pendulum = self.pendulum = DoublePendulum(**self.pendulum_config)
		self.add(pendulum)
		self.wait()
		pendulum.start_swinging()
		self.wait(self.run_time)

class Example1(Example):
	CONFIG = {
		"pendulum_config": {
			"top_point": 2*UP,
			"L1": 2,
			"L2": 2,
			"m1": 1,
			"m2": 1,
			"initial_theta1": 17*DEGREES,
			"initial_theta2": 100*DEGREES,
			"initial_omega1": 0,
			"initial_omega2": 0,
			"weight_diameter": 0.35,
			"gravity": 20,
			"n_steps_per_frame": 100,

			"weight1_color": RED,
			"weight2_color": BLUE,
		},
		"run_time": 25,
	}

class ExampleDouble(Scene):
	CONFIG = {
		"pendulums_config": [
			{
				"top_point": 2*UP,
				"L1": 2,
				"L2": 2,
				"m1": 1,
				"m2": 1,
				"initial_theta1": 17*DEGREES,
				"initial_theta2": 100*DEGREES,
				"initial_omega1": 0,
				"initial_omega2": 0,
				"weight_diameter": 0.35,
				"gravity": 20,
				"n_steps_per_frame": 100,

				"weight1_color": RED,
				"weight2_color": BLUE,
			}, {
				"top_point": 2*UP,
				"L1": 2,
				"L2": 2,
				"m1": 1,
				"m2": 1,
				"initial_theta1": -80*DEGREES,
				"initial_theta2": 25*DEGREES,
				"initial_omega1": 0,
				"initial_omega2": 0,
				"weight_diameter": 0.35,
				"gravity": 20,
				"n_steps_per_frame": 100,

				"weight1_color": ORANGE,
				"weight2_color": GREEN,
			},
		],
		"run_time": 25,
	}
	def construct(self):
		self.pendulums = [DoublePendulum(**config) for config in self.pendulums_config]
		self.add(*self.pendulums)
		self.wait()
		for p in self.pendulums:
			p.start_swinging()
		self.wait(self.run_time)

class ExampleDouble1(ExampleDouble):
	CONFIG = {
		"pendulums_config": [
			{
				"top_point": 2*UP,
				"L1": 2,
				"L2": 2,
				"m1": 1,
				"m2": 1,
				"initial_theta1": 17*DEGREES,
				"initial_theta2": 100*DEGREES,
				"initial_omega1": 0,
				"initial_omega2": 0,
				"weight_diameter": 0.35,
				"gravity": 20,
				"n_steps_per_frame": 3,

				"weight1_color": RED,
				"weight2_color": BLUE,
			}, {
				"top_point": 2*UP,
				"L1": 2,
				"L2": 2,
				"m1": 1,
				"m2": 1,
				"initial_theta1": 17*DEGREES,
				"initial_theta2": 110*DEGREES,
				"initial_omega1": 0,
				"initial_omega2": 0,
				"weight_diameter": 0.35,
				"gravity": 20,
				"n_steps_per_frame": 3,

				"weight1_color": RED,
				"weight2_color": GREEN,
			}, {
				"top_point": 2*UP,
				"L1": 2,
				"L2": 2,
				"m1": 1,
				"m2": 1,
				"initial_theta1": 17*DEGREES,
				"initial_theta2": 120*DEGREES,
				"initial_omega1": 0,
				"initial_omega2": 0,
				"weight_diameter": 0.35,
				"gravity": 20,
				"n_steps_per_frame": 3,

				"weight1_color": RED,
				"weight2_color": ORANGE,
			}, {
				"top_point": 2*UP,
				"L1": 2,
				"L2": 2,
				"m1": 1,
				"m2": 1,
				"initial_theta1": 17*DEGREES,
				"initial_theta2": 130*DEGREES,
				"initial_omega1": 0,
				"initial_omega2": 0,
				"weight_diameter": 0.35,
				"gravity": 20,
				"n_steps_per_frame": 3,

				"weight1_color": RED,
				"weight2_color": PINK,
			},
		],
		"run_time": 25,
	}
