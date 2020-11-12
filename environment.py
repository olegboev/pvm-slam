import cv2
import json
import numpy as np

# Expected length of a wall segment
WALL_SEGMENT_EXPECTED_LENGTH = 30


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
        :param vertex1: first vertex coordinates of a wall line as list
        :param vertex2: second vertex coordinates of a wall line as list
        """
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.segments = Wall._generate_segments(vertex1, vertex2)

    @staticmethod
    def _generate_segments(vertex1, vertex2):
        """ Generates segments with random length and colors.
        :param vertex1: first vertex coordinates of a line as list
        :param vertex2: second vertex coordinates of a line as list
        :return: list of segments
        """
        np.random.seed(11)

        length = np.linalg.norm(np.array(vertex1) - np.array(vertex2))
        segments = list()
        painted_length = 0
        while painted_length < length:
            segment_length = np.random.normal(WALL_SEGMENT_EXPECTED_LENGTH, WALL_SEGMENT_EXPECTED_LENGTH / 5)
            segment_color = np.random.randint(0, 255, 3, dtype=np.int)
            # Convert color type to int for correct drawing by OpenCV
            segment_color = tuple(int(channel) for channel in segment_color)

            painted_length_new = min(painted_length + segment_length, length)
            segments.append(Segment(painted_length, painted_length_new, segment_color))
            painted_length = painted_length_new

        return segments


class Map:
    """ Map is a collection of walls. """

    def __init__(self, map_file_path):
        """ Constructs map by loading its description from a JSON file.
        :param map_file_path: path to JSON file with environment map description.
        """
        self.walls = Map._load_map_data_from_file(map_file_path)

    @staticmethod
    def _load_map_data_from_file(map_file_path):
        """ Loads map walls from a JSON file.
        :param map_file_path: path to JSON file with map description
        :return: map walls
        """

        with open(map_file_path) as json_file:
            data = json.load(json_file)

        return Map._load_map_data(data)

    @staticmethod
    def _load_map_data(map_data):
        """ Loads map data from dictionary.
        :param map_data: dictionary with map data
        :return: map walls
        """
        walls = list()
        for wall in map_data['map']['walls']:
            walls.append(Wall(wall[0:2], wall[2:4]))

        return walls


class Environment:
    """ Class representing an environment. """

    def __init__(self, map_file_path):
        """ Constructs Environment object.
        :param map_file_path: path to JSON file with environment map description.
        """
        self.map = Map(map_file_path)

    def get_image(self):
        """ Returns an image with the environment
        :return: image with environment
        """

        # Calculate image size based on wall coordinates
        vertex_coordinates = np.squeeze(np.array([[w.vertex1 + w.vertex2] for w in self.map.walls]))
        vertex_x_coordinates = vertex_coordinates[:, np.array([0, 2])].flatten()
        vertex_y_coordinates = vertex_coordinates[:, np.array([1, 3])].flatten()

        # Image margin size in px
        margin = 40

        # Assuming all coordinates are positive
        image_width = np.max(vertex_x_coordinates) + margin
        image_height = np.max(vertex_y_coordinates) + margin

        # White image
        image = np.ones([image_height, image_width, 3], dtype=np.uint8) * 255

        for wall in self.map.walls:
            Environment._draw_wall(image, wall, line_width=2)

        return image

    @staticmethod
    def _draw_wall(image, wall, line_width=1):
        """ Draws wall line on the image.
        :param image: image to draw on
        :param wall: wall to draw
        :param line_width: line width to draw
        :return:
        """

        # Unit direction of wall line
        wall_unit_dir = np.array(wall.vertex2) - np.array(wall.vertex1)
        wall_unit_dir = wall_unit_dir / np.linalg.norm(wall_unit_dir)

        for segment in wall.segments:
            points = [tuple(np.round(np.array(wall.vertex1) + wall_unit_dir * t).astype(np.int))
                      for t in [segment.t1, segment.t2]]
            cv2.line(image, points[0], points[1], segment.color, line_width)
