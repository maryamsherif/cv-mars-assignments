import matplotlib.pyplot as plt
from PIL import Image

def calculate_histogram(image_path):
   
    im = Image.open(image_path)
    gray_im = im.convert("L")
    histogram = [0] * 256

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            pixel_value = gray_im.getpixel((i, j))
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



image_path = "D:/Desktop/assign/cv-mars-assign1/test2.jpeg"

#Calculate the histogram
histogram = calculate_histogram(image_path)
print("Histogram (1D array):", histogram)

#Calculate the cumulative histogram
cumulative_histogram = calculate_cumulative_histogram(histogram)
print("Cumulative Histogram:", cumulative_histogram)

#Calculate color intensity at a certain percentage
percentage = 5
lower_intensity, upper_intensity = get_color_at_percentage(cumulative_histogram, percentage)
print(f"Color Intensity at {percentage}%: {lower_intensity} (5%) and {upper_intensity} (95%)")

#Plot the histogram
plt.figure(figsize=(8, 6))
plt.bar(range(256), histogram, width=1.0, color='gray')
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Plot the cumulative histogram
plt.figure(figsize=(8, 6))
plt.plot(range(256), cumulative_histogram, color='black')
plt.title('Cumulative Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Cumulative Frequency')
plt.show()