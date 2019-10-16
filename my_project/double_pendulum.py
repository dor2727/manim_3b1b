from big_ol_pile_of_manim_imports import *

OUTPUT_DIRECTORY = "double_pendulum"

class DoublePendulum(VGroup):
	CONFIG = {
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
		"rod_style": {
			"stroke_width": 3,
			"stroke_color": LIGHT_GREY,
			"sheen_direction": UP,
			"sheen_factor": 1,
		},
		"weight_style": {
			"stroke_width": 0,
			"fill_opacity": 1,
			"fill_color": GREY_BROWN,
			"sheen_direction": UL,
			"sheen_factor": 0.5,
			"background_stroke_color": BLACK,
			"background_stroke_width": 3,
			"background_stroke_opacity": 0.5,
		},
		"weight1_color": RED,
		"weight2_color": BLUE,
		"dashed_line_config": {
			"num_dashes": 25,
			"stroke_color": WHITE,
			"stroke_width": 2,
		},
		"angle_arc_config": {
			"radius": 1,
			"stroke_color": WHITE,
			"stroke_width": 2,
		},
		"include_theta_label": False,
		"include_velocity_vector": False,
		"n_steps_per_frame": 100,
	}
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.initialize()
		if self.include_theta_label:
			# self.add_theta_label()
			pass
		if self.include_velocity_vector:
			# self.add_velocity_vector()
			pass
		self.set_theta1(self.initial_theta1)
		self.set_theta2(self.initial_theta2)

	def initialize(self):
		self.create_fixed_point()
		self.create_rod1()
		self.create_weight1()
		self.create_rod2()
		self.create_weight2()
		self.rotating_group1 = VGroup(self.rod1, self.weight1)
		self.rotating_group2 = VGroup(self.rod2, self.weight2)
		self.create_dashed_line1()
		self.create_dashed_line2()
		self.create_angle_arc1()
		self.create_angle_arc2()
		self.set_omega1(self.initial_omega1)
		self.set_omega2(self.initial_omega2)

	def create_fixed_point(self):
		self.fixed_point_tracker = VectorizedPoint(self.top_point)
		self.add(self.fixed_point_tracker)
		return self
	def get_fixed_point(self):
		return self.fixed_point_tracker.get_location()
	def get_pivot_point(self):
		return self.rod1.get_end()
		return self.weight1.get_center()
		return self.rod2.get_start()

	def _create_rod(self, length, top_point):
		rod = Line(UP, DOWN)
		rod.set_height(length)
		rod.set_style(**self.rod_style)
		rod.move_to(top_point, aligned_edge=UP)
		self.add(rod)
		return rod
	def create_rod1(self):
		self.rod1 = self._create_rod(self.L1, self.get_fixed_point())
	def create_rod2(self):
		self.rod2 = self._create_rod(self.L2, self.get_pivot_point())
	
	def _create_weight(self, center):
		weight = Circle()
		weight.set_width(self.weight_diameter)
		weight.set_style(**self.weight_style)
		weight.move_to(center)
		self.add(weight)
		return weight
	def create_weight1(self):
		self.weight1 = self._create_weight(self.rod1.get_end())
		self.weight1.set_color(self.weight1_color)
	def create_weight2(self):
		self.weight2 = self._create_weight(self.rod2.get_end())
		self.weight2.set_color(self.weight2_color)

	def create_dashed_line1(self):
		self.dashed_line1 = line = DashedLine(
			self.get_fixed_point(),
			self.get_fixed_point() + self.L1 * DOWN,
			**self.dashed_line_config
		)
		line.add_updater(
			lambda l: l.move_to(self.get_fixed_point(), aligned_edge=UP)
		)
		self.add_to_back(line)
	def create_dashed_line2(self):
		self.dashed_line2 = line = DashedLine(
			self.get_pivot_point(),
			self.get_pivot_point() + self.L2 * DOWN,
			**self.dashed_line_config
		)
		line.add_updater(
			lambda l: l.move_to(self.get_pivot_point(), aligned_edge=UP)
		)
		self.add_to_back(line)

	def create_angle_arc1(self):
		self.angle_arc1 = always_redraw(lambda: Arc(
			arc_center=self.get_fixed_point(),
			start_angle=-90 * DEGREES,
			angle=self.get_theta1(),
			**self.angle_arc_config,
		))
		self.add(self.angle_arc1)
	def create_angle_arc2(self):
		self.angle_arc2 = always_redraw(lambda: Arc(
			arc_center=self.get_pivot_point(),
			start_angle=-90 * DEGREES,
			angle=self.get_theta2(),
			**self.angle_arc_config,
		))
		self.add(self.angle_arc2)

	#
	def _get_theta(self, rod, dashed_line):
		theta = rod.get_angle() - dashed_line.get_angle()
		theta = (theta + PI) % TAU - PI
		return theta
	def get_theta1(self):
		return self._get_theta(self.rod1, self.dashed_line1)
	def get_theta2(self):
		return self._get_theta(self.rod2, self.dashed_line2)

	def set_theta1(self, theta):
		self.rotating_group1.rotate(
			theta - self.get_theta1()
		)
		self.rotating_group1.shift(
			self.get_fixed_point() - self.rod1.get_start(),
		)
		return self
	def set_theta2(self, theta):
		self.rotating_group2.rotate(
			theta - self.get_theta2()
		)
		self.rotating_group2.shift(
			self.get_pivot_point() - self.rod2.get_start(),
		)
		return self
	
	def get_omega1(self):
		return self.omega1
	def get_omega2(self):
		return self.omega2
	def set_omega1(self, omega):
		self.omega1 = omega
		return self
	def set_omega2(self, omega):
		self.omega2 = omega
		return self

	def get_energy_detailed(self):
		h1 = 0 # todo
		h2 = 0 # todo
		U_1 = self.m1 * self.gravity * h1
		U_2 = self.m2 * self.gravity * h2

		I_1 = self.m1 * self.L1**2
		I_2 = self.m2 * self.L2**2
		E_k_1 = 0.5 * I_1 self.get_omega1()**2
		E_k_2 = 0.5 * I_2 self.get_omega2()**2

		return {"U1": U_1, "U2": U_2, "EK1": E_k_1, "EK2": E_k_2}
	def get_energy(self):
		return sum(self.get_energy_detailed().values())

	#
	def add_trajectory(self, weight, color=None):
		def update_trajectory(traj, dt):
			new_point = traj.weight.get_center()
			if get_norm(new_point - traj.points[-1]) > 0.01:
				traj.add_smooth_curve_to(new_point)

		traj = VMobject()
		traj.set_color(color or weight.get_color())
		traj.weight = weight
		# traj.start_new_path(p.point)
		traj.start_new_path(weight.get_center())
		traj.set_stroke(weight.get_color(), 1, opacity=0.75)
		traj.add_updater(update_trajectory)
		self.add(traj, weight)
		return traj

	#
	def start_swinging(self):
		self.add_updater(DoublePendulum.update_by_gravity)
		self.traj1 = self.add_trajectory(self.weight1)
		self.traj2 = self.add_trajectory(self.weight2)

	def end_swinging(self):
		self.remove_updater(DoublePendulum.update_by_gravity)

	def update_by_gravity(self, dt):
		m1    , m2     = self.m1          , self.m2
		L1    , L2     = self.L1          , self.L2
		theta1, theta2 = self.get_theta1(), self.get_theta2()
		omega1, omega2 = self.get_omega1(), self.get_omega2()
		sin   , cos    = np.sin           , np.cos
		g              = self.gravity

		M = m1+m2
		delta_theta = theta1 - theta2

		n = self.n_steps_per_frame
		d_dt = dt / n

		for _ in range(n):
			omega_2_dot = (
				(
					(m2**2 / M) * L2 * omega2**2 * sin(delta_theta) * cos(delta_theta)
					 +
					m2 * g * sin(theta1) * cos(delta_theta)
					 +
					m2 * L1 * omega1**2 * sin(delta_theta)
					 -
					m2 * g * sin(theta2)
				)
				 /
				(
					m2*L2
					 -
					(m2**2 / M) * L2 * cos(delta_theta)**2
				)
			)

			omega_1_dot = -(
				(
					m2 * L2 * omega_2_dot * cos(delta_theta)
					 +
					m2 * L2 * omega2**2 * sin(delta_theta)
					 +
					g * M * sin(theta1)
				)
				 /
				(M * L1)
			)

			theta1 += omega1 * d_dt
			theta2 += omega2 * d_dt
			omega1 += omega_1_dot * d_dt
			omega2 += omega_2_dot * d_dt
			# print(omega_1_dot, omega_2_dot)
			# print(theta1, theta2, omega1, omega2)
		self.set_theta1(theta1)
		self.set_theta2(theta2)
		self.set_omega1(omega1)
		self.set_omega2(omega2)


class Pendulum(VGroup):
	CONFIG = {
		"length": 3,
		"gravity": 9.8,
		"weight_diameter": 0.5,
		"initial_theta": 0.3,
		"omega": 0,
		"damping": 0.1,
		"top_point": 2 * UP,
		"rod_style": {
			"stroke_width": 3,
			"stroke_color": LIGHT_GREY,
			"sheen_direction": UP,
			"sheen_factor": 1,
		},
		"weight_style": {
			"stroke_width": 0,
			"fill_opacity": 1,
			"fill_color": GREY_BROWN,
			"sheen_direction": UL,
			"sheen_factor": 0.5,
			"background_stroke_color": BLACK,
			"background_stroke_width": 3,
			"background_stroke_opacity": 0.5,
		},
		"dashed_line_config": {
			"num_dashes": 25,
			"stroke_color": WHITE,
			"stroke_width": 2,
		},
		"angle_arc_config": {
			"radius": 1,
			"stroke_color": WHITE,
			"stroke_width": 2,
		},
		"velocity_vector_config": {
			"color": RED,
		},
		"theta_label_height": 0.25,
		"set_theta_label_height_cap": False,
		"n_steps_per_frame": 100,
		"include_theta_label": True,
		"include_velocity_vector": False,
		"velocity_vector_multiple": 0.5,
		"max_velocity_vector_length_to_length_ratio": 0.5,
	}

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.create_fixed_point()
		self.create_rod()
		self.create_weight()
		self.rotating_group = VGroup(self.rod, self.weight)
		self.create_dashed_line()
		self.create_angle_arc()
		if self.include_theta_label:
			self.add_theta_label()
		if self.include_velocity_vector:
			self.add_velocity_vector()

		self.set_theta(self.initial_theta)
		self.update()

	def create_fixed_point(self):
		self.fixed_point_tracker = VectorizedPoint(self.top_point)
		self.add(self.fixed_point_tracker)
		return self

	def create_rod(self):
		rod = self.rod = Line(UP, DOWN)
		rod.set_height(self.length)
		rod.set_style(**self.rod_style)
		rod.move_to(self.get_fixed_point(), UP)
		self.add(rod)

	def create_weight(self):
		weight = self.weight = Circle()
		weight.set_width(self.weight_diameter)
		weight.set_style(**self.weight_style)
		weight.move_to(self.rod.get_end())
		self.add(weight)

	def create_dashed_line(self):
		line = self.dashed_line = DashedLine(
			self.get_fixed_point(),
			self.get_fixed_point() + self.length * DOWN,
			**self.dashed_line_config
		)
		line.add_updater(
			lambda l: l.move_to(self.get_fixed_point(), UP)
		)
		self.add_to_back(line)

	def create_angle_arc(self):
		self.angle_arc = always_redraw(lambda: Arc(
			arc_center=self.get_fixed_point(),
			start_angle=-90 * DEGREES,
			angle=self.get_arc_angle_theta(),
			**self.angle_arc_config,
		))
		self.add(self.angle_arc)

	def get_arc_angle_theta(self):
		# Might be changed in certain scenes
		return self.get_theta()

	def add_velocity_vector(self):
		def make_vector():
			omega = self.get_omega()
			theta = self.get_theta()
			mvlr = self.max_velocity_vector_length_to_length_ratio
			max_len = mvlr * self.rod.get_length()
			vvm = self.velocity_vector_multiple
			multiple = np.clip(
				vvm * omega, -max_len, max_len
			)
			vector = Vector(
				multiple * RIGHT,
				**self.velocity_vector_config,
			)
			vector.rotate(theta, about_point=ORIGIN)
			vector.shift(self.rod.get_end())
			return vector

		self.velocity_vector = always_redraw(make_vector)
		self.add(self.velocity_vector)
		return self

	def add_theta_label(self):
		self.theta_label = always_redraw(self.get_label)
		self.add(self.theta_label)

	def get_label(self):
		label = TexMobject("\\theta")
		label.set_height(self.theta_label_height)
		if self.set_theta_label_height_cap:
			max_height = self.angle_arc.get_width()
			if label.get_height() > max_height:
				label.set_height(max_height)
		top = self.get_fixed_point()
		arc_center = self.angle_arc.point_from_proportion(0.5)
		vect = arc_center - top
		norm = get_norm(vect)
		vect = normalize(vect) * (norm + self.theta_label_height)
		label.move_to(top + vect)
		return label

	#
	def get_theta(self):
		theta = self.rod.get_angle() - self.dashed_line.get_angle()
		theta = (theta + PI) % TAU - PI
		return theta

	def set_theta(self, theta):
		self.rotating_group.rotate(
			theta - self.get_theta()
		)
		self.rotating_group.shift(
			self.get_fixed_point() - self.rod.get_start(),
		)
		return self

	def get_omega(self):
		return self.omega

	def set_omega(self, omega):
		self.omega = omega
		return self

	def get_fixed_point(self):
		return self.fixed_point_tracker.get_location()

	#
	def start_swinging(self):
		self.add_updater(Pendulum.update_by_gravity)

	def end_swinging(self):
		self.remove_updater(Pendulum.update_by_gravity)

	def update_by_gravity(self, dt):
		theta = self.get_theta()
		omega = self.get_omega()
		nspf = self.n_steps_per_frame
		for x in range(nspf):
			d_theta = omega * dt / nspf
			d_omega = op.add(
				-self.damping * omega,
				-(self.gravity / self.length) * np.sin(theta),
			) * dt / nspf
			theta += d_theta
			omega += d_omega
		self.set_theta(theta)
		self.set_omega(omega)
		return self



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
