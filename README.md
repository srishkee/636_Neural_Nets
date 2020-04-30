Topic: Cutting vegetables

#### How to Reproduce:

1. Individual Images

Download the gui/ folder, and the .h5 file containing the weights. Run “python Cooking_GUI.py” to run the GUI. (Alternatively, the .ipynb file can be run in Google Colab/Jupyter). The GUI allows classification of individual images.

2. Videos

Download the predict.py script, and the .h5 file containing the weights. Specify the file location of the input images, then run “python predict.py” to run the predictor. Note: script assumes the video is already split up into frames. 

3. Demo Link

This link directs to the GUI. Note that the 1st classification will take some time, as the model needs to setup. (This GUI remains a work in progress.): https://youtu.be/Y-2u3SsR188

Please note that I did not include the actual `.jpg` files, as they were too large. Instead, the `.mp4` videos are posted, along with custom python scripts to split them into individual frames.

#### File Explanation:

`Open-Pose-Keras:` The library I used to annotate the images. Cloned from: https://github.com/rachit2403/Open-Pose-Keras

`data:` Contains input data (in video form; later split into frames)

`data_test:` Contains the 5 test videos, as per the project submission guidelines

`gui:` Contains scripts required for the GUI

`outputs:` Contains the log scripts for the multiple model runs. This folder is not strictly relevant to understanding the project.

`plots:` Contains the accuracy/loss plots for the model

`MyJob.LSF:` Contains the job script used for model training. Model was trained on Tamu HPRC (Ada).

`classify.py:` Script to build and train model. Includes utility functions for plotting & saving training data.

`cooking2.h5:` Contains the trained model weights

`data.csv:` Contains accuracy/loss data from training

`predict.py:` Script to get prediction results for a given image/series of images. Generates the required .JSON files and graph. For best results, please use the same folder structure as used in `data_test.` Use the `splitvideos_testing.py` to split the desired video into frames.

`splitvideos_testing.py:` Used to split test videos into frames, at 1 frame/sec. Note: all frames will be resized to (640, 480) since that is the input size of the model

`splitvideos*.py:` Used for splitting training data. (2 different scripts were written to reflect the different folder hierarchies used in both datasets).

`Report:` Contains the project report in both Word and PDF form.

In case of any issues encountered while reproducing the code, please reach out to me and I will be more than happy to help!
2
