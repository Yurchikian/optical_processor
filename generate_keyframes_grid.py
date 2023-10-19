import math
import cv2
import argparse
import numpy as np
import imutils

def get_args_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument('--input', '-i', default='video.mp4', type=str,
                      help='input video file')
  parser.add_argument('--grid_width', '-gw', default=4, type=int,
                      help='width of the image grid')
  parser.add_argument('--grid_height', '-gh', default=4, type=int,
                      help='height of the image grid')
  parser.add_argument('--image_width', '-iw', default=2048, type=int,
                      help='width of the resulting image')
  
  return parser


def read_frame(cap, frame_number):
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
  _ret, frame = cap.read()

  assert _ret

  return frame


def main(args):
  cap = cv2.VideoCapture(args.input)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  print("Total frames = ", total_frames)

  total_keyframes = args.grid_width * args.grid_height
  single_width = args.image_width // args.grid_width

  image = np.empty((0))

  for i in range(args.grid_height):
    row = np.empty((0))

    for j in range(args.grid_width):
      keyframe_index = i * args.grid_width + j

      frame_index = math.floor((total_frames - 1) / (total_keyframes - 1) * keyframe_index)
      frame = read_frame(cap, frame_index)

      resized_frame = imutils.resize(frame, width=single_width)

      if row.any():
        row = np.concatenate([row, resized_frame], axis=1)
      else:
        row = resized_frame
    
    if image.any():
      image = np.concatenate([image, row], axis=0)
    else:
      image = row

  # cv2.imshow('FRAME', image)
  # cv2.waitKey(0)

  cv2.imwrite('out.jpg', image)


if __name__ == '__main__':
  parser = get_args_parser()
  args = parser.parse_args()
  main(args)
