import math
import cv2
import argparse
import numpy as np
import imutils
import pathlib
from tqdm import tqdm

import torch
import torchvision
import torchfields
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def get_file_name(path):
  path = pathlib.Path(path)
  return path.name


def read_frame(cap, frame_number, target_shape):
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
  _ret, frame = cap.read()

  assert _ret

  return imutils.resize(frame, width=target_shape[1], height=target_shape[0])

def get_grid_cell(grid, i, image_shape, grid_shape):
  grid_cell_position = (i // grid_shape[0], i % grid_shape[0])
  grid_cell_offset = (grid_cell_position[0] * image_shape[0], grid_cell_position[1] * image_shape[1])

  # print(grid.shape, grid_cell_offset)

  image = grid[grid_cell_offset[0]:grid_cell_offset[0] + image_shape[0],
              grid_cell_offset[1]:grid_cell_offset[1] + image_shape[1]]

  return pad_to_multiple_of(image)


def get_args_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument('--input', '-i', default='video.mp4', type=str,
                      help='input video file')
  parser.add_argument('--input_grid', '-ig', default=None, type=str,
                      help='input image file')
  parser.add_argument('--grid_width', '-gw', default=4, type=int,
                      help='width of the image grid')
  parser.add_argument('--grid_height', '-gh', default=4, type=int,
                      help='height of the image grid')
  
  return parser


def pad(size, multiple=8):
  # print("SIZE: ", size, ",", math.ceil(size / multiple), " -> ", math.ceil(size / multiple) * multiple)

  return math.ceil(size / multiple) * multiple


def pad_to_multiple_of(image, multiple=8):
  width, height, depth = image.shape
  
  desired_width = pad(width, multiple)
  desired_height = pad(height, multiple)

  image = cv2.copyMakeBorder(image, 0, desired_width - width, 0, desired_height - height, borderType=cv2.BORDER_REPLICATE)

  # print("new shape: ", image.shape)
  return image


def main(args):
  cap = cv2.VideoCapture(args.input)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # print(args)

  grid_image = cv2.imread(args.input_grid)

  height, width, _channels = grid_image.shape
  image_size = (height // args.grid_height, width // args.grid_width)
  # image_size = (pad(image_size[0]), pad(image_size[1]))
  total_keyframes = args.grid_width * args.grid_height

  assert total_keyframes >= 2 

  keyframes = []
  root_frames_indices = []
  root_frames = []

  for i in range(total_keyframes):
    keyframes.append(get_grid_cell(grid_image, i, image_size, (args.grid_width, args.grid_height)))

    root_frame_index = math.floor((total_frames - 1) / (total_keyframes - 1) * i)
    root_frames_indices.append(root_frame_index)
    root_frames.append(read_frame(cap, root_frame_index, image_size))


  print(root_frames_indices)

  for i in tqdm(range(total_frames)):
    keyframe_index, progress = get_neighbor_keyframe_indices(i + 1, total_frames, total_keyframes)
    
    # print("Processing frame #%05d" % i)
    output_directory = pathlib.Path(get_file_name(args.input_grid) + '_out')
    output_directory.mkdir(exist_ok=True)

    if i in root_frames_indices:
      image = keyframes[keyframe_index]
      # print("Frame %05d is a keyframe %05d" % (i, root_frames_indices[keyframe_index]))
    else:
      this_frame = read_frame(cap, i, image_size)
      prev_root_frame = root_frames[keyframe_index]
      next_root_frame = root_frames[keyframe_index + 1]


      prev_image = keyframes[keyframe_index]
      next_image = keyframes[keyframe_index + 1]
      
      prev_image_warped = apply_flow_as(prev_image, prev_root_frame, this_frame)
      next_image_warped = apply_flow_as(next_image, next_root_frame, this_frame)

      # prev_image_warped = prev_image
      # next_image_warped = next_image

      # print("Frame %05d" % i, "prev_keyframe_id", keyframe_index, progress)
      image = prev_image_warped * (1 - progress) + next_image_warped * progress
    
    output_path = output_directory.joinpath("frame_%05d.jpg" % i).as_posix()

    # print(output_path)
    cv2.imwrite(output_path, image)
    # print(i, root_frame, progress)
  
  command = ['ffmpeg', '-y', '-i',
            output_directory.joinpath('frame_%05d.jpg').as_posix(), 
            '-r', '%s' % cap.get(cv2.CAP_PROP_FPS), "%s.mp4" % output_directory.as_posix()]

  import subprocess
  subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def prepare_model():
  model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device)
  return model.eval()


def get_flow(a, b):
  a = pad_to_multiple_of(a)
  b = pad_to_multiple_of(b)

  a = converter(a).to(device)
  b = converter(b).to(device)

  return model(b[None], a[None])[-1]


def apply_flow(image, flow):
  image_tensor = converter(image).to(device)
  displacement_field = flow.field().from_pixels()
  # print("displacement_field. From ", displacement_field.min().item(), " to ", displacement_field.max().item())

  displaced_image = (displacement_field)(image_tensor).cpu().detach().numpy() * 255
  displaced_image = np.moveaxis(displaced_image, 0, 2)

  return displaced_image


def apply_flow_as(image, ref_from, ref_to):
  optical_flow = get_flow(ref_from, ref_to)
  image = apply_flow(image, optical_flow)

  return image

  
def get_neighbor_keyframe_indices(frame, total_frames, total_keyframes):
  keyframe = (frame / (total_frames - 1)) * (total_keyframes - 1)
  keyframe_progress = keyframe - math.floor(keyframe)


  progress_sigmoid_factor = 3.9
  progress = 1 / (1 + math.pow(math.e, -(2 * progress_sigmoid_factor * keyframe_progress - progress_sigmoid_factor)))
  # progress = keyframe_progress
  
  if keyframe == (total_keyframes - 1):
    keyframe -= 1
    keyframe_progress = 1.0

  return (math.floor(keyframe), progress)



if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = prepare_model()
  converter = torchvision.transforms.ToTensor()
  parser = get_args_parser()
  args = parser.parse_args()

  if args.input_grid == None:
    args.input_grid = "%s.jpg" % get_file_name(args.input)

  main(args)
