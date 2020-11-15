import json

import cv2
import numpy as np

from camera import Camera
from environment import Environment


def main():
    map_data_file = r'map.json'
    environment = Environment.load_from_file(map_data_file)

    # TODO refactor visualization

    trajectory = [[100, 100, -90], [700, 100, -90],
                  [700, 100, -180], [700, 500, -180],
                  [700, 500, -270], [100, 500, -270],
                  [100, 500, -360], [100, 100, -360],
                  [100, 100, -450]]

    for i in range(len(trajectory) - 1):
        j = i + 1
        value1 = np.array(trajectory[i], dtype=np.float)
        value2 = np.array(trajectory[j], dtype=np.float)

        value1[-1] = np.deg2rad(value1[-1])
        value2[-1] = np.deg2rad(value2[-1])

        steps = 20
        for t in range(steps):

            value = value1 + t * (value2 - value1) / steps

            camera_position = tuple(value[:2])
            camera_yaw = value[2]

            camera = Camera(environment, 40, (40, 1), camera_position, camera_yaw)
            image_camera = camera.make_shot()

            image_environment = environment.get_image(camera)

            scale = 15
            image_camera = cv2.resize(image_camera, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            image_result = np.ones((900, 1000, 3), dtype=np.uint8) * 255
            cv2.putText(image_result, 'Map', (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2, cv2.LINE_AA)
            image_result[50:50 + image_environment.shape[0], 100:100 + image_environment.shape[1], :] = \
                image_environment

            cv2.putText(image_result, 'Image on camera', (350, 100 + image_environment.shape[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2, cv2.LINE_AA)

            image_result[780:780+image_camera.shape[0], 230:230+image_camera.shape[1], :] = image_camera

            cv2.imshow('map', image_result)

            k = cv2.waitKey(2)
            if 27 == k:
                exit()


if __name__ == '__main__':
    main()
