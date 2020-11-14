import json

import cv2

from environment import Environment


def main():

    map_data_file = r'map.json'
    environment = Environment.load_from_file(map_data_file)
    image = environment.get_image()

    cv2.imshow('map', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()