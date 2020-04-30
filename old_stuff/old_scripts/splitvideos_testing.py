"""
This script splits a video into frames at rate 1 frame/sec. Used to split the 5 TEST VIDEOS.

Note: ONLY EDIT "FOLDER_NAME" !!!

Processed folders:

Images_Cutting_Folders/
1. door_cut_ch0/ 
2. window_cut_ch0/
3. sink_cut_ch0/
4. fridge_cut_ch0/

"""

import cv2, os
print(cv2.__version__)

vidcap = cv2.VideoCapture()

FOLDER_NAME = 'test_vid_4' # ONLY EDIT THIS!!!

folder_dir = '/scratch/user/skumar55/cooking/data_test/' + FOLDER_NAME + '/'
output_dir = '/scratch/user/skumar55/cooking/data_test/' + FOLDER_NAME + '/test_imgs/'
output_dir = output_dir
print(output_dir)

print(os.listdir('.'))
videos = os.listdir(folder_dir)
print("Videos: ", videos)

# REMOVE!!!!!
videos = [FOLDER_NAME + '.mp4']

imgs=os.listdir(output_dir)
print("Before: ", len(imgs))

RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

# Loop over all videos in folder & split them into frames
for i,video in enumerate(videos):
  print(video)
  vidcap.open(folder_dir+video)
  
  # Get frame rate
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  print("FPS: ", fps)

  count = 0
  success = 1
  while(success):
    success, img = vidcap.read() # Get frame
    if(success):
      # frame_name = output_dir + video[:-4] + "_video" + str(i) + "_" + "frame" + str(count) + ".jpg"
      frame_name = output_dir + str(count) + ".jpg"
      print(frame_name)
      resized_img = img
      resized_img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation = cv2.INTER_LINEAR) # Resize to match video size of (480x640)
      cv2.imwrite(frame_name, resized_img) # Write frame to Images/
      count += 1
    else: break

    # Skip "fps" frames to reach 1frame/sec
    ctr = 0
    while(ctr < fps): # Want fps=1 frame/2 sec
      success, img = vidcap.read()
      ctr+=1

  print("Successfully parsed video#", i, "into", count, "frames!")
  # if(i==0): break

print('Finished!')

imgs=os.listdir(output_dir)
print("After: ", len(imgs))

