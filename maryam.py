# Modify histogram
# In this part, you are asked to implement two functions as follows:
# 1. StretchContrast
    # • Input: 2D array representing the image (feel free to use a predefined function to transform an image into an array), and four numbers representing color intensities.
    # • Output: An image presenting the effect of the contrast stretching on the input image.
    # • Description: Implement contrast stretching as discussed in class.

# 2. EqualizeHistogram
    # • Input: 2D array representing the image (feel free to use a predefined function to transform an image into an array), 
    # and two numbers representing color intensities.
    # • Output: An image presenting the effect of the histogram equalization on the input image.
    # • Description: Implement histogram equalization as discussed in class 
    # with the modification of tuning the linearization as per the frequency of the given color intensities.

# 3. Gray-scaleTransformation
    # • Input: 2D array representing the image (feel free to use a predefined function to transform an image into an array), and four numbers representing color intensities.
    # • Output: An image presenting the effect of the gray-scale transformation on the input image.
    # • Description: Implement gray-scale transformation as discussed in class.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import cv2
from PIL import Image
import numpy as np

# Open an image
image = Image.open('test/cv_tut.png')

# Convert the image to a NumPy array
img = np.array(image)

def StretchContrast(img, a, b, c, d):

    #check if there are outlying pixels and if yes, then choose c and d accordingly
    #if no, then c and d are the min and max of the image
    # if np.any(img < c):
    #     c = np.min(img)
    
    scaling_factor = (b-a)/(d-c)

    for i in range(len(img)): #rows
        for j in range(len(img[i])): #columns

            
            # checking if the pixel values are within the range [c, d]
            if np.all([c <= img[i][j], img[i][j] <= d]):
                #if yes, then apply the formula
                img[i][j] = ((img[i][j]-c)*scaling_factor)+a

    #show the image
    plt.imshow(img)
    plt.show()
    return img

    

            

def EqualizeHistogram(img, a, b):
    return 0
    

def GrayScaleTransformation(img, a, b, c, d):
    return 0

#main function
if __name__ == "__main__":
    resulting_image = StretchContrast(img, 0, 255, 88, 151)
    #EqualizeHistogram(img, 0, 255)
    #GrayScaleTransformation(img, 0, 255, 0, 255)
    pass
