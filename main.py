# USAGE
# python main.py -d yihuojia16A -f yihuojia
import argparse
import imutils
import cv2
import csv
import os
import numpy as np
from pathlib import Path


def calculate_bgr_mean(image):
  return cv2.mean(image)[:3]


def calculate_lab_mean(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.mean(image)[:3]
  return l * 100 / 255, a - 128, b - 128


def init_argparse():
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--dir', required=True,
                  help='Directory containing images')
  ap.add_argument('-f', '--factory', required=True, help='Factory name')
  ap.add_argument(
      '-w', '--width', help='Set the width of the image (Default to 1500)', default=1500)
  return ap


def get_roi(image, window_name):
  x0, y0, dx, dy = cv2.selectROI(
      window_name, image, showCrosshair=False)
  x1 = x0 + dx
  y1 = y0 + dy
  return image[y0:y1, x0:x1]


def filter_jpg_files(files_list):
  return list(filter(lambda name: name.endswith('.jpg'), files_list))


def get_args_info(ap):
  args = vars(ap.parse_args())
  return args['dir'], args['factory'], args['width'], os.listdir(args['dir'])


def main():
  ap = init_argparse()
  directory, factory, width, files = get_args_info(ap)
  jpgs = filter_jpg_files(files)

  with open('card_info.csv', 'w+', newline='') as csv_file:
    fieldnames = ['card_id', 'card_factory', 'b',
                  'g', 'r', 'l_star', 'a_star', 'b_star']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for jpg in jpgs:
      path = str(Path(directory, jpg))
      image = cv2.imread(path)
      resized = imutils.resize(image, width=width)
      roi = get_roi(resized, window_name=path)

      b, g, r = calculate_bgr_mean(roi)
      l_star, a_star, b_star = calculate_lab_mean(roi)
      writer.writerow({'card_id': 'null_id', 'card_factory': factory, 'b': b,
                       'g': g, 'r': r, 'l_star': l_star, 'a_star': a_star, 'b_star': b_star})

    csv_file.close()


if __name__ == '__main__':
  main()
