
Before detecting corner, we need to apply [[Grayscaling]] and covert to float type:
```python
# Convert to grayscale
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

# Detect corners 
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate corner image to enhance corner points
dst = cv2.dilate(dst,None)

plt.imshow(dst, cmap='gray')
# First param after the image is the pixel size of the neighbourhood to look at
# Second param is the pixel size of the sobel kernels used (usually 3)
# Thurd param: a constant - typically 0.04 (low number means more corners to be detected)
```
```python
## TODO: Define a threshold for extracting strong corners
# This value vary depending on the image and how many corners you want to detect
# Try changing this free parameter, 0.1, to be larger or smaller ans see what happens
thresh = 0.1*dst.max()

# Create an image copy to draw corners on
corner_image = np.copy(image_copy)

# Iterate through all the corners and draw them on the image (if they pass the threshold)
for j in range(0, dst.shape[0]):
    for i in range(0, dst.shape[1]):
        if(dst[j,i] > thresh):
            # image, center pt, radius, color, thickness
            cv2.circle( corner_image, (i, j), 1, (0,255,0), 1)

plt.imshow(corner_image)
```
Similiar to [[edge detection]], corner detection is based on high intensity in the image.

But here not only the magnitude matters (the strength of the change in intensity) but also the direction of the change.

When we shift a kernel through the image, detections of BIG VARIANCE IN MAGNITUDE AND DIRECTION MEAN CORNER !

To obtain the magnitude and the direction, we calculate the intensity gradient in the x and y direction by applying [[Sobel Filters]].


![[Pasted image 20210820115538.png|500]]

Then we can calculate the magnitude and the direction by converting the gradients x and y into polar coordinates as follows:

![[Pasted image 20210820115959.png|500]]