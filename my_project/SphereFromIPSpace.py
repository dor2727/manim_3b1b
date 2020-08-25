class SphereFromIPSpace(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(70*DEGREES, 300*DEGREES)
        self.camera.frame_center.shift(UR)
        
        plane = NumberPlane(
            background_line_style = {"stroke_color": GREY},
            x_axis_config = {"x_min":-1, "x_max": 5, "stroke_color":MAROON},
            y_axis_config = {"x_min":-1, "x_max": 8, "stroke_color":GREEN}
        )
        
        dx, dy = 0.2, 0.2
        sphere_rects = VGroup()
        
        for x in np.arange(0, PI, dx):
            for y in np.arange(0, 2*PI, dy):
                rect = Rectangle(width=dx, height=dy, fill_opacity=0.6)\
                rect.shift((x+dx/2)*RIGHT + (y+dy/2)*UP)
                sphere_rects.add(rect)
                
        sphere_rects.set_color_by_gradient(BLUE, GREEN, MAROON)
        self.add(plane, sphere_rects)
        self.wait()

        def param_func(point):
            u, v = point[:-1]
            return np.array([
                np.cos(v) * np.sin(u),
                np.sin(v) * np.sin(u),
                np.cos(u)
            ])

        self.play(
            LaggedStart(
                *[ApplyMethod(mob.apply_function, param_func, lag_ratio=0.5)
                for mob in sphere_rects]
            ),
            run_time = 8
        )
        self.wait()