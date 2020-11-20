import json

import cv2
import numpy as np

from camera import Camera
from detector import Detector
from environment import Environment
from frame import Frame
from view import View


def main():
    map_data_file = r'map.json'
    environment = Environment.load_from_file(map_data_file)
    camera = Camera(environment, 30, (50, 1), (0, 0), 0)
    detector = Detector()
    matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2, crossCheck=True)
    view = View(environment, camera)

    trajectory = [[100, 100, -90], [700, 100, -90],
                  [700, 100, -180], [700, 300, -180],
                  [700, 300, -270], [100, 300, -270],
                  [100, 300, -360], [100, 100, -360],
                  [100, 100, -450]]

    frame_prev = None
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

            camera_image = camera.get_frame_image()
            kp, des = detector.detect_and_compute(camera_image)
            frame_curr = Frame(camera_image, kp, des)

            matches = list()
            if frame_curr.descriptors and frame_prev and frame_prev.descriptors:
                matches = matcher.match(np.array(frame_prev.descriptors), np.array(frame_curr.descriptors))

            view.frame_prev = frame_prev
            view.frame_curr = frame_curr
            view.matches = matches

            image_result = view.draw()
            frame_prev = frame_curr

            cv2.imshow('map', image_result)

            k = cv2.waitKey(20)
            if 27 == k:
                exit()


if __name__ == '__main__':
    main()
