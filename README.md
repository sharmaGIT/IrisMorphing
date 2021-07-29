#
# Iris Morphing
Code for creating iris morphs from two iris images belong to two different subjects.

# Requirement
Numpy, Scipy, OpenCV, Matplotlib

# Description
Iris morphing creates a synthetic (morphed) iris image comprising images from two different subjects. It consists of three steps: landmark detection, warping and blending. More details on these steps could be found in the research paper specified below.

<img src="https://github.com/sharmaGIT/IrisMorphing/blob/main/Images/IrisMorphing_Figure.jpg" width="800" height="400">

# Running of the code
The main function is in the irismorphing.py python file. To create iris morphs, there are five arguments: path of image1, path of image2, segmentation information of image1, segmentation information of image2, and path of output morphed image. The segmentation information should be in the form of [iriscenterX, iriscenterY, irisradius, pupilcenterX, pupilcenterY, pupilradius]. Two test images are provided in the "Images" folder. A morphed image from the test images could be created by directly running irismorphing.py without arguments as default arguments are given in the python file. The output morphed image would be generated in the same "Images" folder.


# Citation
If you are using the code, please cite the paper:

Renu Sharma, Arun Ross, IMAGE-LEVEL IRIS MORPH ATTACK, IEEE International Conference on Image Processing (ICIP), 2021.
