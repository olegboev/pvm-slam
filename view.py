import cv2
import numpy as np


class View:
    """ View class is used for displaying the state of the environment. """

    def __init__(self, environment, camera):
        """ View constructor.
        :param environment: environment object
        :param camera: camera object
        """

        self.environment = environment
        self.camera = camera

    def draw(self):
        """ Creates an image with the current state.
        :return: image with current state.
        """
        image_environment = self._draw_environment()
        image_camera_frame = self.camera.get_frame_image()
        image_result = self._compose_result_image(image_environment, image_camera_frame)

        return image_result

    def _compose_result_image(self, image_environment, image_frame):
        """
        Draws all the components on the result image.
        :param image_environment: image of the environment
        :param image_frame: image of the camera frame
        :return: rusult image composed of environment and frame images
        """

        # Draw current state
        scale = 15
        image_camera = cv2.resize(image_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        image_result = np.ones((900, 1000, 3), dtype=np.uint8) * 255
        cv2.putText(image_result, 'Map', (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2, cv2.LINE_AA)
        image_result[50:50 + image_environment.shape[0], 100:100 + image_environment.shape[1], :] = \
            image_environment
        cv2.putText(image_result, 'Image on camera', (350, 100 + image_environment.shape[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2, cv2.LINE_AA)
        image_result[780:780 + image_camera.shape[0], 230:230 + image_camera.shape[1], :] = image_camera
        return image_result

    def _draw_environment(self):
        """ Returns an image with the environment
        :return: image with environment
        """

        # Image margin size in px
        margin = 40

        vertex_x_coordinates = np.array([[w.vertex1[0], w.vertex2[0]] for w in self.environment.map.walls])
        vertex_y_coordinates = np.array([[w.vertex1[1], w.vertex2[1]] for w in self.environment.map.walls])

        # Assuming all coordinates are positive
        image_width = np.max(vertex_x_coordinates) + margin
        image_height = np.max(vertex_y_coordinates) + margin

        # White image
        image = np.ones([image_height, image_width, 3], dtype=np.uint8) * 255

        # Draw walls
        for wall in self.environment.map.walls:
            View._draw_wall(image, wall, thickness=2)

        # Draw camera
        self._draw_camera_symbol(image, thickness=2)

        return image

    @staticmethod
    def _draw_wall(image, wall, thickness=1):
        """ Draws wall line on the image.
        :param image: image to draw on
        :param wall: wall to draw
        :param thickness: line thickness
        :return:
        """

        # Unit direction of wall line
        wall_unit_dir = wall.vertex2 - wall.vertex1
        wall_unit_dir = wall_unit_dir / np.linalg.norm(wall_unit_dir)

        for segment in wall.segments:
            points = [tuple(np.round(wall.vertex1 + wall_unit_dir * t).astype(np.int))
                      for t in [segment.t1, segment.t2]]
            cv2.line(image, points[0], points[1], segment.color, thickness)

    def _draw_camera_symbol(self, image, thickness=1):
        """ Draws camera symbol on a map image.
        :param image: image where camera will be drawn
        :param camera: object of Camera class
        :param thickness: line thickness
        :return:
        """
        # Currently yaw=0 is aligned with negative direction of Y axis
        # Vertices of a triangle representing camera
        vertices = np.zeros((4, 3))
        vertices[:, 0] = [0, 0, -30, 1]
        vertices[:, 1] = [-10, 0, 0, 1]
        vertices[:, 2] = [10, 0, 0, 1]

        vertices_plane = (self.camera.C2W @ vertices)[:2, :]

        num_lines = vertices_plane.shape[1]
        for i in range(num_lines):
            j = (i + 1) % num_lines
            cv2.line(image, tuple(np.round(vertices_plane[:, i]).astype(np.int)),
                     tuple(np.round(vertices_plane[:, j]).astype(np.int)), (0, 0, 0), thickness)
