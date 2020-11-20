import cv2
import numpy as np

FRAME_SCALE = 15

class View:
    """ View class is used for displaying the state of the environment. """

    def __init__(self, environment, camera):
        """ View constructor.
        :param environment: environment object
        :param camera: camera object
        """

        self.environment = environment
        self.camera = camera
        self.frame_prev = None
        self.frame_curr = None
        self.matches = None

    def draw(self):
        """ Creates an image with the current state.
        :return: image with current state.
        """
        image_environment = self._draw_environment()
        image_camera_frames = self._draw_camera_frames()
        image_result = self._compose_result_image(image_environment, image_camera_frames)

        return image_result

    def _draw_camera_frames(self):
        """
        Draws frames captured by camera and keypoint matches.
        :return: Image with frames and matches.
        """

        image_frame_curr_resized = cv2.resize(self.frame_curr.image, None, fx=FRAME_SCALE, fy=FRAME_SCALE,
                                       interpolation=cv2.INTER_NEAREST)
        if self.frame_prev:
            image_frame_prev_resized = cv2.resize(self.frame_prev.image, None, fx=FRAME_SCALE, fy=FRAME_SCALE,
                                       interpolation=cv2.INTER_NEAREST)
            kp1_resized = [cv2.KeyPoint((kp.pt[0] + 0.5) * FRAME_SCALE, kp.pt[1] * FRAME_SCALE, kp.size)
                           for kp in self.frame_prev.keypoints]
        else:
            image_frame_prev_resized = np.zeros_like(image_frame_curr_resized)
            kp1_resized = list()

        kp2_resized = [cv2.KeyPoint((kp.pt[0] + 0.5) * FRAME_SCALE, kp.pt[1] * FRAME_SCALE, kp.size)
                       for kp in self.frame_curr.keypoints]

        for kp in kp1_resized:
            cv2.circle(image_frame_prev_resized, tuple(np.round(kp.pt).astype(np.int)), 4, (200, 200, 200), 2)
        for kp in kp2_resized:
            cv2.circle(image_frame_curr_resized, tuple(np.round(kp.pt).astype(np.int)), 4, (200, 200, 200), 2)

        image_frames = np.vstack([image_frame_prev_resized, image_frame_curr_resized])

        for match in self.matches:
            p1 = np.array(kp1_resized[match.queryIdx].pt)
            p2 = np.array(kp2_resized[match.trainIdx].pt)
            p1_coord = tuple(np.round(p1).astype(np.int))
            p2_coord = tuple(np.round(p2 + [0, image_frame_prev_resized.shape[0]]).astype(np.int))
            cv2.line(image_frames, p1_coord, p2_coord, (200, 200, 200), 2)

        return image_frames

    def _compose_result_image(self, image_environment, image_frames):
        """ Draws all the components on the result image.
        :param image_environment: image of the environment
        :param image_frames: image of the camera frames
        :return: result image composed of environment and frame images
        """

        # Draw current state
        image_result = np.ones((700, 1000, 3), dtype=np.uint8) * 255

        cv2.putText(image_result, 'Map', (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2, cv2.LINE_AA)
        image_result[50:50 + image_environment.shape[0], 100:100 + image_environment.shape[1], :] = \
            image_environment

        cv2.putText(image_result, 'Previous/current frames and keypoint matches', (150, 100 + image_environment.shape[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2, cv2.LINE_AA)
        image_result[580:580 + image_frames.shape[0], 140:140 + image_frames.shape[1], :] = image_frames

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
