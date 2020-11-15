import unittest
import numpy as np

from geometry import intersect_ray_segment


class TestGeometry(unittest.TestCase):
    """ Tests for Geometry utility functions """

    def test_intersect_ray_segment_intersect(self):
        """ Test for intersect_ray_segment() function when there is an intersection.
        :return:
        """
        p1 = np.array([0, 0])
        p2 = np.array([4, 0])
        q1 = np.array([5, -1])
        q2 = np.array([5, 3])
        u = intersect_ray_segment(p1, p2, q1, q2)
        self.assertEqual(u, 0.25)

    def test_intersect_ray_segment_not_intersect(self):
        """ Test for intersect_ray_segment() function when there is no intersection.
        :return:
        """
        p1 = np.array([0, 0])
        p2 = np.array([4, 0])
        q1 = np.array([5, 1])
        q2 = np.array([5, 3])
        u = intersect_ray_segment(p1, p2, q1, q2)
        self.assertIsNone(u)

    def test_intersect_ray_segment_parallel(self):
        """ Test for intersect_ray_segment() function when a ray and a line are parallel.
        :return:
        """
        p1 = np.array([0, 0])
        p2 = np.array([4, 0])
        q1 = np.array([1, 1])
        q2 = np.array([5, 1])
        u = intersect_ray_segment(p1, p2, q1, q2)
        self.assertIsNone(u)

    def test_intersect_ray_segment_collinear(self):
        """ Test for intersect_ray_segment() function when a ray and a line are collinear.
        :return:
        """
        p1 = np.array([0, 0])
        p2 = np.array([4, 0])
        q1 = np.array([5, 0])
        q2 = np.array([7, 0])
        u = intersect_ray_segment(p1, p2, q1, q2)
        self.assertIsNone(u)


if __name__ == '__main__':
    unittest.main()
