import json

import cv2
import numpy as np

from camera import Camera
from environment import Environment
from view import View


def main():
    map_data_file = r'map.json'
    environment = Environment.load_from_file(map_data_file)
    camera = Camera(environment, 40, (40, 1), (0, 0), 0)

    view = View(environment, camera)

    trajectory = [[100, 100, -90], [700, 100, -90],
                  [700, 100, -180], [700, 500, -180],
                  [700, 500, -270], [100, 500, -270],
                  [100, 500, -360], [100, 100, -360],
                  [100, 100, -450]]

    for i in range(len(trajectory) - 1):
        j = i + 1
        trajectory_point_start = np.array(trajectory[i], dtype=np.float)
        trajectory_point_finish = np.array(trajectory[j], dtype=np.float)

        trajectory_point_start[-1] = np.deg2rad(trajectory_point_start[-1])
        trajectory_point_finish[-1] = np.deg2rad(trajectory_point_finish[-1])

        steps = 20
        for t in range(steps):

            trajectory_point = trajectory_point_start + t * (trajectory_point_finish - trajectory_point_start) / steps

            camera.position = tuple(trajectory_point[:2])
            camera.yaw = trajectory_point[2]

            image_result = view.draw()

            cv2.imshow('map', image_result)

            k = cv2.waitKey(2)
            if 27 == k:
                exit()


if __name__ == '__main__':
    main()
