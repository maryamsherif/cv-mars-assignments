import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

              
def calculate_histogram(image_array):
   
    histogram = [0] * 256

    for i in range(len(image_array)):
        for j in range(len(image_array[0])):
            pixel_value = image_array[i][j]
            histogram[pixel_value] += 1

    return histogram


def calculate_cumulative_histogram(histogram):
    cumulative_histogram = [0] * len(histogram)
    cumulative_histogram[0] = histogram[0]

    for i in range(1, len(histogram)):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]

    return cumulative_histogram


def get_color_at_percentage(cumulative_histogram, percentage):
    total_pixels = cumulative_histogram[-1]  
    target_value = percentage * total_pixels / 100 
    
    for i in range(len(cumulative_histogram)):
        if cumulative_histogram[i] >= target_value:
            lower_index = i
            break

    for i in range(len(cumulative_histogram) - 1, -1, -1):
        if cumulative_histogram[i] <= (total_pixels - target_value):
            upper_index = i
            break

    
    lower_intensity = lower_index
    upper_intensity = upper_index

    return lower_intensity, upper_intensity

#------------------------------------------------------------------------------------------------
def StretchContrast(img, a, b, c, d):
    scaling_factor = (b - a) / (d - c)

    for i in range(len(img)):  # rows
        for j in range(len(img[i])):  # columns
            pixel_value = img[i][j]
            if c <= pixel_value <= d:
                img[i][j] = int(((pixel_value - c) * scaling_factor) + a)

            # Clip the values to the range [a, b] which is [0,255] if there are outlying pixel values    
            elif pixel_value < c:
                img[i][j] = a  # Clip to 'a' for values below 'c'
            else:
                img[i][j] = b  # Clip to 'b' for values above 'd'

    return img
        
    
def EqualizeHistogram(img, lower_intensity, higher_intensity):
    # Step 1: Calculate Histogram
    histogram = calculate_histogram(img)

    # Step 2: Calculate Cumulative Histogram (CDF)
    cdf = calculate_cumulative_histogram(histogram)

    # Step 3: Apply histogram equalization within the specified intensity range
    for i in range(len(img)):
        for j in range(len(img[i])):
            pixel_value = img[i][j]
            if lower_intensity <= pixel_value <= higher_intensity:
                img[i][j] = int((cdf[pixel_value] - cdf[lower_intensity]) * (255 / (cdf[higher_intensity] - cdf[lower_intensity])))

            # Clip the values to the range [0, 255] for outlying pixel values
            elif pixel_value < lower_intensity:
                img[i][j] = 0  # Clip to 0 for values below the lower_intensity
            else:
                img[i][j] = 255  # Clip to 255 for values above the higher_intensity

    return img



def GrayScaleTransformation(img, x1, x2, y1, y2):

    if x1 >= x2 or y1 >= y2:
        print("Error: x1 must be less than x2 and y1 must be less than y2")
        return img
    
    for i in range(len(img)):
        for j in range (len(img[i])):
            pixel_value = img[i][j]
            if pixel_value < x1:
                img[i][j] = int (pixel_value * (y1/x1))
            elif x1 <= pixel_value <= x2:
                img[i][j] = int(((pixel_value - x1) * ((y2-y1)/(x2-x1))) + y1)
            else:
                img[i][j] = int(((pixel_value - x2) * ((255-y2)/(255-x2))) + y2)

    return img

#main function
if __name__ == "__main__":

    # Enter the image path
    image_path = "test2.png"

    im = Image.open(image_path)
    gray_scale = im.convert("L")
    image_array = [[gray_scale.getpixel((i, j)) for j in range(im.size[1])] for i in range(im.size[0])]

    #Calculate the histogram
    histogram = calculate_histogram(image_array)
    #print("Histogram (1D array):", histogram)

    #Calculate the cumulative histogram
    cumulative_histogram = calculate_cumulative_histogram(histogram)
    #print("Cumulative Histogram:", cumulative_histogram)

    #Calculate color intensity at a certain percentage
    percentage = 5
    lower_intensity, upper_intensity = get_color_at_percentage(cumulative_histogram, percentage)
    print("Lower Intensity:", lower_intensity)
    print("Upper Intensity:", upper_intensity)

    # # Stretch Contrast: 
    # stretched_image = StretchContrast(image_array, 0, 255, 88, 150)
    # stretched_image = np.transpose(stretched_image)
    # plt.subplot(121), plt.imshow(gray_scale, cmap='gray'), plt.title('Original Image')
    # plt.subplot(122), plt.imshow(stretched_image, cmap='gray'), plt.title('Stretched Image')
    # plt.show()
    
    # #Equalize Histogram:
    # equalized_image = EqualizeHistogram(image_array, lower_intensity, upper_intensity)
    # equalized_image = np.transpose(equalized_image)
    # plt.subplot(121), plt.imshow(gray_scale, cmap='gray'), plt.title('Original Image')
    # plt.subplot(122), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized Image')
    # plt.show()

    # #Gray Scale Transformation:
    # transformed_image = GrayScaleTransformation(image_array, lower_intensity, upper_intensity, 10, 200)
    # transformed_image = np.transpose(transformed_image)
    # plt.subplot(121), plt.imshow(gray_scale, cmap='gray'), plt.title('Original Image')
    # plt.subplot(122), plt.imshow(transformed_image, cmap='gray'), plt.title('Transformed Image')
    # plt.show()

    pass
