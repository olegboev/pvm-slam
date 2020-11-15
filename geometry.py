import numpy as np


def yaw_to_rotation_matrix(yaw):
    """
    Calculates rotation matrix from yaw angle rotation about world Z coordinate axis.
    :param yaw: yaw angle in radians
    :return: 3x3 rotation matrix
    """

    sin = np.sin(yaw)
    cos = np.cos(yaw)

    R_z = np.array([[cos, sin, 0],
                    [-sin, cos, 0],
                    [0, 0, 1]])

    return R_z


def intersect_ray_segment(p1, p2, q1, q2):
    """
    Calculates intersection between a ray and a line segment.
    :param p1: ray beginning point as np.array
    :param p2: point on ray as np.array
    :param q1: segment beginning point as np.array
    :param q2: segment ending point as np.array
    :return: parameter value within segment, or None if there is no intersection or a ray and a segment are collinear
    or parallel
    """

    # Algorithm: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    r = p2 - p1
    s = q2 - q1
    s_cross_r = np.cross(s, r)

    if np.isclose(s_cross_r, 0):
        return None

    u = np.cross((p1 - q1), r) / s_cross_r
    
    if not 0 <= u <= 1:
        return None

    return u
