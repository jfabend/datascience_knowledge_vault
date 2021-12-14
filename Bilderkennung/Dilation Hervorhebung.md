Dilation enlarges bright, white areas in an image by adding pixels to the perceived boundaries of objects in that image. Hence, it is the opposite of [[Erosion]].

![[Pasted image 20210820143546.png|300]]

Often performed on binary images

```python
# Reads in a binary image
image = cv2.imread(‘j.png’, 0) 

# Create a 5x5 kernel of ones
kernel = np.ones((5,5),np.uint8)

# Dilate the image
dilation = cv2.dilate(image, kernel, iterations = 1)
```