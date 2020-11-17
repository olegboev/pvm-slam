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

