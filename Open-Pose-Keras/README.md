# Open-Pose-Keras

Cloned from: https://github.com/rachit2403/Open-Pose-Keras

While the original repo provides support for 19 body points, I modified `demo_image.py` to omit body points for the hips, knees, and ankles, as only the upper body posture is actually relevant. (For instance - sitting/standing while cutting vegetables does not make a signficant difference in posture). This lead to cleaner, more interpretable image annotations.


<div align="center">
<img src="https://github.tamu.edu/skumar55/636-Neural-Nets/blob/master/Report/all_points.jpg", width="300", height="300">
&nbsp;
<img src="https://github.tamu.edu/skumar55/636-Neural-Nets/blob/master/Report/12_points.jpg", width="300", height="300">
</div>


Utility scripts (not actually used by classification/prediction scripts, but only for image annotation purposes):

`demo_image.py:` Modified script that annotates a folder of images. (Input = folder path)

`resize_images.py:` Script to resize images. 

`move_files.py:` Script to move files/folders around.