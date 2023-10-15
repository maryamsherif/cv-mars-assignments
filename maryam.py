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
    #1- calculate the histogram
    #2- calculate the cumulative histogram
    #3- calculate the color intensity at a certain percentage
    #4- apply the formula
    #5- show the image

    
    return 0
    

def GrayScaleTransformation(img, a, b, c, d):
    return 0

#main function
if __name__ == "__main__":
    resulting_image = StretchContrast(img, 0, 255, 88, 151)
    #EqualizeHistogram(img, 0, 255)
    #GrayScaleTransformation(img, 0, 255, 0, 255)
    pass
