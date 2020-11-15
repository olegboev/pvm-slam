import cv2
import json
import numpy as np

# Expected length of a wall segment
WALL_SEGMENT_EXPECTED_LENGTH = 30

np.random.seed(11)

class Segment:
    """ Segment is a part of a wall with the constant color. """

    def __init__(self, t1, t2, color):
        """ Constructs segment.
        :param vertex1: parameter value of the first segment vertex on a wall line
        :param vertex2: parameter value of the second segment vertex on a wall line
        :param color: segment color
        """
        self.t1 = t1
        self.t2 = t2
        self.color = color


class Wall:
    """ Wall is a line with segments of different colors. """

    def __init__(self, vertex1, vertex2):
        """ Constructs wall.
        :param vertex1: first vertex coordinates of a wall line as np.array([x, y])
        :param vertex2: second vertex coordinates of a wall line as np.array([x, y])
        """
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.segments = Wall._generate_segments(vertex1, vertex2)

    def get_color_at(self, t):
        """ Returns color of the wall at the point defined by parameter value t, which starts at vertex1.
        :param t: parameter, must be between 0 and 1 inclusive.
        :return: color of the wall at the point defined by parameter value.
        """
        assert 0 <= t <= 1

        point_location = t * np.linalg.norm(self.vertex1 - self.vertex2)

        for segment in self.segments:
            if segment.t1 <= point_location <= segment.t2:
                return segment.color

        raise ValueError(f'Unable to find segment for parameter {t}')

    @staticmethod
    def _generate_segments(vertex1, vertex2):
        """ Generates segments with random length and colors.
        :param vertex1: first vertex coordinates of a line as np.array([x, y])
        :param vertex2: second vertex coordinates of a line as np.array([x, y])
        :return: list of segments
        """

        length = np.linalg.norm(vertex1 - vertex2)
        segments = list()
        painted_length = 0
        while painted_length < length:
            segment_length = np.random.normal(WALL_SEGMENT_EXPECTED_LENGTH, WALL_SEGMENT_EXPECTED_LENGTH / 5)

            color_hsv = np.array([[[np.random.randint(0, 179), 255, 255]]]).astype(np.uint8)
            segment_color = tuple(cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0, :])
            # Convert color type to int for correct usage by OpenCV
            segment_color = tuple(int(channel) for channel in segment_color)

            painted_length_new = min(painted_length + segment_length, length)
            segments.append(Segment(painted_length, painted_length_new, segment_color))
            painted_length = painted_length_new

        return segments


class Map:
    """ Map is a collection of walls. """

    def __init__(self, map_data):
        """ Constructs map by loading its description from a JSON file.
        :param map_data: dictionary with map data
        """
        self.walls = Map._load_wall_data(map_data)

    @staticmethod
    def _load_wall_data(map_data):
        """ Loads map data from dictionary.
        :param map_data: dictionary with map data
        :return: map walls
        """
        walls = list()
        prev_vertex = None
        for vertex in map_data['map']['vertices']:
            if prev_vertex is None:
                prev_vertex = vertex
            else:
                walls.append(Wall(np.array(prev_vertex), np.array(vertex)))
                prev_vertex = vertex

        return walls


class Environment:
    """ Class representing an environment. """

    def __init__(self, map_data):
        """ Constructs Environment object.
        :param map_data: dictionary with map data
        """
        self.map = Map(map_data)

    @staticmethod
    def load_from_file(map_file_path):
        """ Loads environment data from JSON file and constructs Environment object from it.
        :param map_file_path: path to JSON file with map description
        :return: Environment object
        """

        with open(map_file_path) as json_file:
            data = json.load(json_file)

        return Environment(data)

    def get_image(self, camera):
        """ Returns an image with the environment
        :param camera: camera which position is used to draw camera symbol on a map
        :return: image with environment
        """

        vertex_x_coordinates = np.array([[w.vertex1[0], w.vertex2[0]] for w in self.map.walls])
        vertex_y_coordinates = np.array([[w.vertex1[1], w.vertex2[1]] for w in self.map.walls])

        # Image margin size in px
        margin = 40

        # Assuming all coordinates are positive
        image_width = np.max(vertex_x_coordinates) + margin
        image_height = np.max(vertex_y_coordinates) + margin

        # White image
        image = np.ones([image_height, image_width, 3], dtype=np.uint8) * 255

        # Draw walls
        for wall in self.map.walls:
            Environment._draw_wall(image, wall, thickness=2)

        # Draw camera
        Environment._draw_camera(image, camera, thickness=2)

        return image

    @staticmethod
    def _draw_camera(image, camera, thickness=1):
        """ Draws camera symbol on map image.
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

        vertices_plane = (camera.C2W @ vertices)[:2, :]

        num_lines = vertices_plane.shape[1]
        for i in range(num_lines):
            j = (i + 1) % num_lines
            cv2.line(image, tuple(np.round(vertices_plane[:, i]).astype(np.int)),
                     tuple(np.round(vertices_plane[:, j]).astype(np.int)), (0, 0, 0), thickness)

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
