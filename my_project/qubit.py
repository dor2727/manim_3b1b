from manimlib.imports import *

OUTPUT_DIRECTORY = "qubit"

Hadamard = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
Pauli_x = np.array([[0,1],[1,0]])
Pauli_y = np.array([[0,-1j],[1j,0]])
Pauli_z = np.array([[1,0],[0,-1]])
Sqrt_x = 1/2 * np.array([[1+1j,1-1j],[1-1j,1+1j]])
def Phase(phi):
	return np.array([[1,0],[0,np.exp(1j * phi)]])

SPHERE_RADIUS = 2

def angles_to_vector(theta, phi):
	zero = complex( np.cos(theta/2) )
	one = np.exp(1j * phi) * np.sin(theta/2)
	return np.array([zero, one])
def vector_to_angles(v):
	# theta = np.arccos(v[0].real)*2
	theta = np.arccos(2 * abs(v[0])**2 - 1)
	print("theta: ", theta)

	if v[0] == 0 or v[1] == 0 or abs(v[0]) == 1:
		print("    reseting phi")
		phi = 0
	else:
		try:
			phi = -1j * np.log(
				( abs(v[0]) * v[1] )
				 /
				(v[0] * np.sqrt(
					1 - abs(v[0])**2
				))
			)
		except:
			print("    reseting phi (error)")
			phi = 0

	t = theta.real
	p = phi.real
	if p < 0:
		p += 2*PI
	return t, p

	# old phi calculation
	if theta == 0 or np.sin(theta/2) == 0 or v[1] == 0:
		print("    reseting phi")
		phi = 0
	else:
		phi = -1j * np.log(v[1] / np.sin(theta/2))
		if almost_zero(phi.imag):
			phi = phi.real
	print("phi: ", phi)
	return theta.real, phi.real

def almost_zero(d):
	return np.around(d, 5) == 0
def complex_to_str(c):
	if almost_zero(c.imag):
		return str(np.around(c.real, 3))
	if almost_zero(c.real):
		return str(np.around(c.imag, 3)) + 'j'
	return str(np.around(c, 3))
def angle_to_str(c):
	return str(np.around(c / DEGREES, 3))


class State(Mobject):
	def __init__(self, zero_amplitude, one_amplitude, r=SPHERE_RADIUS, **kwargs):
	# def __init__(self, theta=0, phi=0, r=SPHERE_RADIUS, **kwargs):
		Mobject.__init__(self, **kwargs)

		self.zero_amplitude = complex(zero_amplitude)
		self.one_amplitude = complex(one_amplitude)

		self.r = r
		self.theta, self.phi = vector_to_angles(self.get_vector())
		# self.phi = phi
		# self.theta = theta

		self.line = self.create_line()
		self.add(self.line)

	def _get_cartesian(self):
		return np.array( spherical_to_cartesian(self.r, self.theta, self.phi) )

	def create_line(self):
		return Line(
			start=ORIGIN,
			end=self._get_cartesian(),
		)

	def get_vector(self):
		return np.array([self.zero_amplitude, self.one_amplitude])
		# return angles_to_vector(self.theta, self.phi)

	def apply_operator(self, operator):
		print("from: ", self.get_vector())
		vector_result = operator.dot(self.get_vector())
		print("to  : ", vector_result)
		new_state = State(*vector_result)
		new_state.set_color(self.color)
		return new_state

class BlochSphere(SpecialThreeDScene):
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

		"circle_xz_show": False,
		"circle_xz_color": PINK,

		"circle_xy_show": True,
		"circle_xy_color": GREEN,

		"circle_yz_show": False,
		"circle_yz_color": GRAY,

		
		"sphere_config": {
			"radius": SPHERE_RADIUS,
			"resolution": (60, 60),
		},
		
		"operators": [
		],
		"operator_names": [
		],
		"show_intro": True,

		"wait_time": 2,
		"pre_operators_wait_time": 1.5,
		"final_wait_time": 3,
		"intro_wait_time": 3,
		"intro_fadeout_wait_time": 1,
	}

	def construct(self):
		if self.show_intro:
			self.present_introduction()
		self.init_camera()
		self.init_axes()
		self.init_sphere()
		self.init_states()
		self.init_text()
		self.wait(self.pre_operators_wait_time)

		for o in self.operators:
			self.apply_operator(o)
			self.wait(self.wait_time)
		self.wait(self.final_wait_time)

	def present_introduction(self):
		self.intro_tex_1 = TextMobject(
			"\\begin{flushleft}"
			"The state of the Qbit"
			"\\\\"
			"as represented in the Bloch Sphere."
			"\\end{flushleft}",
	        alignment="",
		)
		self.intro_tex_1.move_to(2*UP)
		self.add(self.intro_tex_1)
		self.play(
			Write(self.intro_tex_1),
			run_time=1.5
		)

		self.intro_tex_2 = TextMobject(
			"\\begin{flushleft}"
			"The following gates will be applied:"
			"\\\\"
			+
			"\\\\".join(f"{i+1}) {n}" for i,n in enumerate(self.operator_names))
			+
			"\\end{flushleft}",
	        alignment="",
		)
		self.intro_tex_2.move_to(0.8*DOWN)
		self.add(self.intro_tex_2)
		self.play(
			Write(self.intro_tex_2),
			run_time=2.5
		)
		self.wait(self.intro_wait_time)
		self.play(
			FadeOut(self.intro_tex_1),
			FadeOut(self.intro_tex_2)
		)
		self.wait(self.intro_fadeout_wait_time)

	def init_camera(self):
		self.set_camera_orientation(**self.init_camera_orientation)

	def init_axes(self):
		self.axes = self.get_axes()
		self.set_axes_labels()
		self.add(self.axes)

	def _tex(self, *s):
		tex = TexMobject(*s)
		tex.rotate(90 * DEGREES, RIGHT)
		tex.rotate(90 * DEGREES, OUT)
		tex.scale(0.5)
		return tex

	def set_axes_labels(self):
		labels = VGroup()

		zero = self._tex("\\ket{0}")
		zero.next_to(
			self.axes.z_axis.number_to_point(1),
			Y_AXIS + Z_AXIS,
			MED_SMALL_BUFF
		)

		one = self._tex("\\ket{1}")
		one.next_to(
			self.axes.z_axis.number_to_point(-1),
			Y_AXIS - Z_AXIS,
			MED_SMALL_BUFF
		)

		labels.add(zero, one)
		self.axes.z_axis.add(labels)

		x = self._tex("x")
		x.next_to(
			self.axes.x_axis.number_to_point(1),
			-Y_AXIS,
			MED_SMALL_BUFF
		)
		self.axes.x_axis.add(x)

		y = self._tex("y")
		y.next_to(
			self.axes.y_axis.number_to_point(1),
			Y_AXIS + Z_AXIS,
			MED_SMALL_BUFF
		)
		self.axes.y_axis.add(y)

	def init_sphere(self):
		sphere = self.get_sphere(**self.sphere_config)
		sphere.set_fill(BLUE_E)
		sphere.set_opacity(0.1)
		self.add(sphere)
		self.sphere = sphere

		if self.circle_xy_show:
			self.circle_xy = Circle(
				radius=SPHERE_RADIUS,
				color=self.circle_xy_color,
			)
			self.circle_xy.set_fill(self.circle_xy_color)
			self.circle_xy.set_opacity(0.1)
			self.add(self.circle_xy)

		if self.circle_xz_show:
			self.circle_xz = Circle(
				radius=SPHERE_RADIUS,
				color=self.circle_xz_color,
			)
			self.circle_xz.rotate(90 * DEGREES, RIGHT)
			self.circle_xz.set_fill(self.circle_xz_color)
			self.circle_xz.set_opacity(0.1)
			self.add(self.circle_xz)

		if self.circle_yz_show:
			self.circle_yz = Circle(
				radius=SPHERE_RADIUS,
				color=self.circle_yz_color,
			)
			self.circle_yz.rotate(90 * DEGREES, UP)
			self.circle_yz.set_fill(self.circle_yz_color)
			self.circle_yz.set_opacity(0.1)
			self.add(self.circle_xy)

	def init_text(self):
		"""
		for each state, write (with its own color):
			the probabilities
			theta & phi
		"""
		# self.tex_zero_vec   = self._tex("\\ket{0} = ", "1.0+0.0j", " \\\\ ", "0.0+0.0j")
		self.tex_zero_vec   = self._tex("\\ket{BLUE} = ", "\\qquad \\qquad 1", " \\\\ ", "\\qquad 0")
		self.tex_zero_vec.set_color(BLUE)
		self.tex_zero_vec.move_to(Z_AXIS * 2 - Y_AXIS * 4)

		self.tex_zero_theta = self._tex("\\theta = ", "0.000")
		self.tex_zero_theta.set_color(BLUE)
		self.tex_zero_theta.move_to(Z_AXIS * 1 - Y_AXIS * 4)

		self.tex_zero_phi   = self._tex("\\phi = ", "0.000")
		self.tex_zero_phi.set_color(BLUE)
		self.tex_zero_phi.move_to(Z_AXIS * 0.5 - Y_AXIS * 4)


		self.tex_one_vec    = self._tex("\\ket{RED} = ", "\\qquad \\qquad 0", " \\\\ ", "\\qquad 1")
		self.tex_one_vec.set_color(RED)
		self.tex_one_vec.move_to(Z_AXIS * 2 + Y_AXIS * 3.5)

		self.tex_one_theta  = self._tex("\\theta = ", "180.0")
		self.tex_one_theta.set_color(RED)
		self.tex_one_theta.move_to(Z_AXIS * 1 + Y_AXIS * 4)

		self.tex_one_phi    = self._tex("\\phi = ", "0.000")
		self.tex_one_phi.set_color(RED)
		self.tex_one_phi.move_to(Z_AXIS * 0.5 + Y_AXIS * 4)

		self.tex_dot_product= self._tex("\\bra{0}\\ket{1} = ", "0.000")
		self.tex_dot_product.set_color(WHITE)
		self.tex_dot_product.move_to(- Z_AXIS * 2 + Y_AXIS * 3)

		self.add(
			self.tex_zero_vec,
			self.tex_zero_theta,
			self.tex_zero_phi,

			self.tex_one_vec,
			self.tex_one_theta,
			self.tex_one_phi,

			self.tex_dot_product,
		)

		# the initial values are only used to make enough space for later values
		self.play(
			*self.update_tex_transforms(self.zero, self.one),
			run_time=0.1
		)

	def update_tex_transforms(self, new_zero, new_one):
		zero_state = new_zero.get_vector()
		zero_angles = vector_to_angles(zero_state)
		one_state = new_one.get_vector()
		one_angles = vector_to_angles(one_state)

		dot_product = np.vdot( new_one.get_vector(), new_zero.get_vector())

		def transform(source, dest):
			t = self._tex(dest)
			t.move_to(source.get_center())
			t.set_color(source.get_color())
			return Transform(source, t)

		# self.play(
		# 	Transform(self.tex_zero_vec[1],   self._tex(zero_state[0])),
		# 	Transform(self.tex_zero_vec[3],   self._tex(zero_state[1])),
		# 	Transform(self.tex_zero_theta[1], self._tex(zero_angles[0])),
		# 	Transform(self.tex_zero_phi[1],   self._tex(zero_angles[1])),

		# 	Transform(self.tex_one_vec[1],   self._tex(one_state[0])),
		# 	Transform(self.tex_one_vec[3],   self._tex(one_state[1])),
		# 	Transform(self.tex_one_theta[1], self._tex(one_angles[0])),
		# 	Transform(self.tex_one_phi[1],   self._tex(one_angles[1])),
		# )
		return(
			transform(self.tex_zero_vec[1],   complex_to_str(zero_state[0])),
			transform(self.tex_zero_vec[3],   complex_to_str(zero_state[1])),
			transform(self.tex_zero_theta[1], angle_to_str(zero_angles[0])),
			transform(self.tex_zero_phi[1],   angle_to_str(zero_angles[1])),

			transform(self.tex_one_vec[1],   complex_to_str(one_state[0])),
			transform(self.tex_one_vec[3],   complex_to_str(one_state[1])),
			transform(self.tex_one_theta[1], angle_to_str(one_angles[0])),
			transform(self.tex_one_phi[1],   angle_to_str(one_angles[1])),

			transform(self.tex_dot_product[1],   complex_to_str(dot_product)),
		)

	def init_states(self):
		# self.old_zero = self.zero = State(0, 0, r=2)
		# self.old_one  = self.one  = State(180*DEGREES, 0, r=2)
		self.old_zero = self.zero = State(1, 0, r=2)
		self.old_one  = self.one  = State(0, 1, r=2)

		self.zero.set_color(BLUE)
		self.one.set_color(RED)

		self.add(self.zero, self.one)

	def apply_operator(self, operator):
		print()
		print("00000")
		new_zero = self.zero.apply_operator(operator)
		print("11111")
		new_one = self.one.apply_operator(operator)

		self.play(
			Transform(self.old_zero, new_zero),
			Transform(self.old_one, new_one),
			*self.update_tex_transforms(new_zero, new_one),
		)

		self.zero = new_zero
		self.one = new_one


class BlochSphere_example_X(BlochSphere):
	CONFIG = {
		"operators": [
			Pauli_x,
		],
		"operator_names": [
			"Pauli X",
		],
	}
class BlochSphere_example_Y(BlochSphere):
	CONFIG = {
		"operators": [
			Pauli_y,
		],
		"operator_names": [
			"Pauli Y",
		],
	}
class BlochSphere_example_Z(BlochSphere):
	CONFIG = {
		"operators": [
			Pauli_z,
		],
		"operator_names": [
			"Pauli Z",
		],
	}

class BlochSphere_example_X_X(BlochSphere):
	CONFIG = {
		"operators": [
			Pauli_x,
			Pauli_x,
		],
		"operator_names": [
			"Pauli X",
			"Pauli X",
		],
	}
class BlochSphere_example_Y_Y(BlochSphere):
	CONFIG = {
		"operators": [
			Pauli_y,
			Pauli_y,
		],
		"operator_names": [
			"Pauli Y",
			"Pauli Y",
		],
	}
class BlochSphere_example_Z_Z(BlochSphere):
	CONFIG = {
		"operators": [
			Pauli_z,
			Pauli_z,
		],
		"operator_names": [
			"Pauli Z",
			"Pauli Z",
		],
	}

class BlochSphere_example_H_H(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Hadamard,
		],
		"operator_names": [
			"Hadamard",
			"Hadamard",
		],
	}
class BlochSphere_example_H_Z(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Pauli_z,
		],
		"operator_names": [
			"Hadamard",
			"Pauli Z",
		],
	}
class BlochSphere_example_H_Z_H(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Pauli_z,
			Hadamard,
		],
		"operator_names": [
			"Hadamard",
			"Pauli Z",
			"Hadamard",
		],
	}
class BlochSphere_example_H_X_H(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Pauli_x,
			Hadamard,
		],
		"operator_names": [
			"Hadamard",
			"Pauli X",
			"Hadamard",
		],
	}

class BlochSphere_example_SX_SX(BlochSphere):
	CONFIG = {
		"operators": [
			Sqrt_x,
			Sqrt_x,
		],
		"operator_names": [
			"Sqrt of X",
			"Sqrt of X",
		],
	}
class BlochSphere_example_SX_SX_SX_SX(BlochSphere):
	CONFIG = {
		"operators": [
			Sqrt_x,
			Sqrt_x,
			Sqrt_x,
			Sqrt_x,
		],
		"operator_names": [
			"Sqrt of X",
			"Sqrt of X",
			"Sqrt of X",
			"Sqrt of X",
		],
	}
class BlochSphere_example_SX_P90_SX(BlochSphere):
	CONFIG = {
		"operators": [
			Sqrt_x,
			Phase(90 * DEGREES),
			Sqrt_x,
		],
		"operator_names": [
			"Sqrt of X",
			"Phase 90",
			"Sqrt of X",
		],
	}
class BlochSphere_example_SX_P180_SX(BlochSphere):
	CONFIG = {
		"operators": [
			Sqrt_x,
			Phase(180 * DEGREES),
			Sqrt_x,
		],
		"operator_names": [
			"Sqrt of X",
			"Phase 180",
			"Sqrt of X",
		],
	}
class BlochSphere_example_SX_P270_SX(BlochSphere):
	CONFIG = {
		"operators": [
			Sqrt_x,
			Phase(270 * DEGREES),
			Sqrt_x,
		],
		"operator_names": [
			"Sqrt of X",
			"Phase 270",
			"Sqrt of X",
		],
	}
class BlochSphere_example_SX_P45_SX(BlochSphere):
	CONFIG = {
		"circle_xz_show": True,
		"operators": [
			Sqrt_x,
			Phase(45 * DEGREES),
			Sqrt_x,
		],
		"operator_names": [
			"Sqrt of X",
			"Phase 45",
			"Sqrt of X",
		],
	}
class BlochSphere_example_SX_P45_SX_SX(BlochSphere):
	CONFIG = {
		"circle_xz_show": True,
		"operators": [
			Sqrt_x,
			Phase(45 * DEGREES),
			Sqrt_x,
			Sqrt_x,
		],
		"operator_names": [
			"Sqrt of X",
			"Phase 45",
			"Sqrt of X",
			"Sqrt of X",
		],
	}
class BlochSphere_example_SX_P45_SX_SX_SX_SX(BlochSphere):
	CONFIG = {
		"circle_xz_show": True,
		"operators": [
			Sqrt_x,
			Phase(45 * DEGREES),
			Sqrt_x,
			Sqrt_x,
			Sqrt_x,
			Sqrt_x,
		],
		"operator_names": [
			"Sqrt of X",
			"Phase 45",
			"Sqrt of X",
			"Sqrt of X",
			"Sqrt of X",
			"Sqrt of X",
		],
	}
class BlochSphere_example_SX_P45_SX_Y_SX(BlochSphere):
	CONFIG = {
		"circle_xz_show": True,
		"operators": [
			Sqrt_x,
			Phase(45 * DEGREES),
			Sqrt_x,
			Pauli_y,
			Sqrt_x,
		],
		"operator_names": [
			"Sqrt of X",
			"Phase 45",
			"Sqrt of X",
			"Pauli Y",
			"Sqrt of X",
		],
	}

class BlochSphere_example_H_P180(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Phase(180 * DEGREES),
		],
		"operator_names": [
			"Hadamard",
			"Phase 180",
		],
	}
class BlochSphere_example_P180_H(BlochSphere):
	CONFIG = {
		"operators": [
			Phase(180 * DEGREES),
			Hadamard,
		],
		"operator_names": [
			"Phase 180",
			"Hadamard",
		],
	}
class BlochSphere_example_H_P180_H(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Phase(180 * DEGREES),
			Hadamard,
		],
		"operator_names": [
			"Hadamard",
			"Phase 180",
			"Hadamard",
		],
	}
class BlochSphere_example_H_P90(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Phase(90 * DEGREES),
		],
		"operator_names": [
			"Hadamard",
			"Phase 90",
		],
	}
class BlochSphere_example_H_P90_H(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Phase(90 * DEGREES),
			Hadamard,
		],
		"operator_names": [
			"Hadamard",
			"Phase 90",
			"Hadamard",
		],
	}
class BlochSphere_example_H_P90_H_SX(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Phase(90 * DEGREES),
			Hadamard,
			Sqrt_x,
		],
		"operator_names": [
			"Hadamard",
			"Phase 90",
			"Hadamard",
			"Sqrt of X",
		],
	}
class BlochSphere_example_H_P90_H_H(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Phase(90 * DEGREES),
			Hadamard,
			Hadamard,
		],
		"operator_names": [
			"Hadamard",
			"Phase 90",
			"Hadamard",
			"Hadamard",
		],
	}
class BlochSphere_example_H_P45_H(BlochSphere):
	CONFIG = {
		"operators": [
			Hadamard,
			Phase(45 * DEGREES),
			Hadamard,
		],
		"operator_names": [
			"Hadamard",
			"Phase 45",
			"Hadamard",
		],
	}


