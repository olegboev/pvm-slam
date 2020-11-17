import numpy as np

from geometry import yaw_to_rotation_matrix, intersect_ray_segment


class Camera:
    """ Camera produces images of the environment. """

    def __init__(self, environment, focus, image_size, position, yaw):
        """ Camera constructor.
        :param environment: environment
        :param focus: focal length
        :param image_size: image size in pixels as tuple (width, height)
        :param position: camera center position in world coordinate frame as tuple (width, height)
        :param yaw: camera yaw angle in world coordinate frame
        """
        self.environment = environment
        self._focus = focus
        self._image_size = image_size
        self.position = position
        self.yaw = yaw

        # Intrinsics matrix and its inversion
        self._K = None
        self._K_inv = None

        # Projection matrix
        self._P = None

        # World to camera
        self._W2C = None
        # Camera to world
        self._C2W = None

        # Rotation of camera to world frame in initial camera position
        self.RC2W_init = Camera._calculate_RC2W_init()


    @property
    def focus(self):
        return self._focus

    @property
    def image_size(self):
        return self._image_size

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position
        self._W2C = None
        self._C2W = None

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, yaw):
        self._yaw = yaw
        self._W2C = None
        self._C2W = None

    @property
    def K(self):
        if self._K is None:
            self._K = self._calculate_K()
        return self._K

    @property
    def K_inv(self):
        if self._K_inv is None:
            self._K_inv = np.linalg.inv(self.K)
        return self._K_inv

    @property
    def C2W(self):
        if self._C2W is None:
            self._C2W = self._calculate_C2W()
        return self._C2W


    @property
    def W2C(self):
        if self._W2C is None:
            self._W2C = np.linalg.inv(self.C2W)
        return self._W2C

    @property
    def P(self):
        if self._P is None:
            self._P = self._calculate_P()
        return self._P

    def _calculate_K(self):
        """ Calculates camera intrinsics matrix K.
        :return: matrix K as 3x3 numpy array
        """
        K = np.eye(3, dtype=np.float)
        K[0, 0] = self.focus
        K[1, 1] = self.focus
        K[0, 2] = (self.image_size[0] - 1) * 0.5
        K[1, 2] = (self.image_size[1] - 1) * 0.5

        return K

    @staticmethod
    def _calculate_RC2W_init():
        """ Calculates rotation matrix of camera initial position in the world coordinate frame.
        :return: RC2W as 3x3 numpy array
        """
        # Initially camera X axis is directed right and Z axis is up in the world frame
        R_C2W_init = np.zeros((3, 3))
        R_C2W_init[0, 0] = 1
        R_C2W_init[1, 2] = -1
        R_C2W_init[2, 1] = 1

        return R_C2W_init


    def _calculate_C2W(self):
        """ Calculates rotation and translation matrix of camera in the world coordinate frame.
        :return: C2W as 4x4 numpy array
        """

        R_yaw = yaw_to_rotation_matrix(self.yaw)
        t = np.hstack([self.position, 0])

        Rt = np.eye(4)
        Rt[:3, :3] = R_yaw @ self.RC2W_init
        Rt[:3, 3] = t

        return Rt

    def _calculate_P(self):
        """ Calculates camera projection matrix P.
        :return: P as 3x4 numpy array
        """
        return self.K @ self.W2C[:3, :]

    def _cast_ray(self, point):
        """ Casts a ray from pixel at point coordinates and returns its color.
        :param point: tuple with x, y coordinates of the pixel
        :param P_inv: inverted camera projection matrix
        :return: point color of the nearest wall
        """

        ray_direction_cam_frame = self.K_inv @ np.hstack([point[0], point[1], 1])
        point_on_image_plane_world_frame = self.C2W @ np.hstack([ray_direction_cam_frame, 1])
        point_on_image_plane_world_frame = point_on_image_plane_world_frame / point_on_image_plane_world_frame[3]

        # p1 - point in camera center, p2 - point on image plane
        p1 = self.position
        p2 = point_on_image_plane_world_frame[:2]
        color = (0, 0, 0)
        for wall in self.environment.map.walls:
            # q1, q2 - wall vertices
            q1 = wall.vertex1
            q2 = wall.vertex2
            t = intersect_ray_segment(p1, p2, q1, q2)
            if t is not None:
                # Check that point is in front of the camera
                intersection_point = q1 + t * (q2 - q1)
                direction = np.dot(p2 - p1, intersection_point - p2)
                if direction > 0:
                    color = wall.get_color_at(t)
        return color

    def get_frame_image(self):
        """ Makes a picture of the environment.
        :return: picture as numpy array in BGR color space
        """

        image = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        for x in range(self.image_size[0]):
            for y in range(self.image_size[1]):
                image[y, x] = self._cast_ray((x, y))
        return image
