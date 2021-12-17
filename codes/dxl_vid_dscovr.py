import glob as glob
import numpy as np
import cv2

file_list = np.sort(glob("/home/cephadrius/Dropbox/DXL-Figure/historical/sw_dsco_*.png"))

for filename in file_list:

    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# Start writing the images to videos
fourcc = cv2.VideoWriter_fourcc(*'MPEG')

# Define the frame rate for video
fps = 25

# Define the video name and its extension
vid_name = f"{file_list[0][:-3]}_{file_list[-1][-19:-3]}_{fps}.mp4"

out = cv2.VideoWriter(vid_name, fourcc, fps, size)

for i, xx in enumerate(img_array):
    xxc = np.uint8( xx.copy())
    out.write( xxc )
    print(f'file {i} written')

out.release()
