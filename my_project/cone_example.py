from manim import *

class ConeScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(70*DEGREES, 300*DEGREES)
        self.camera.frame_center.shift(UR)


        a = Arrow3d(-Y_AXIS, 2*Y_AXIS)
        self.add(a)

        cone = Cone(direction=Y_AXIS)
        self.add(cone)

        self.wait()

class ThreeDCone(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        cone = Cone(
            base_radius=0.5,
            direction=-Y_AXIS,
            checkerboard_colors=[RED_D, RED_E], resolution=(15, 32)
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, cone)
        self.wait()

class ThreeDArrow(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        arrow = Arrow3d(
            start=-X_AXIS,
            end=2*X_AXIS,
            checkerboard_colors=[RED_D, RED_E], resolution=(15, 32)
        )
        arrow.shift(1*Z_AXIS)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, arrow)
        self.wait()

class ThreeDParaboloid(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        paraboloid = ParaboloidPolar(
            r_max=1,
            center_point= -Z_AXIS,
        )

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, paraboloid)
        self.wait()

