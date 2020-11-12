import cv2

from environment import Environment


def main():
    environment = Environment('map.json')
    image = environment.get_image()

    cv2.imshow('map', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()