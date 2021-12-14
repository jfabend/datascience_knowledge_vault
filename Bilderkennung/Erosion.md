The opposite of [[Dilation Hervorhebung]]

![[Pasted image 20210820143326.png|300]]
```python
# Reads in a binary image
image = cv2.imread(‘j.png’, 0) 

# Create a 5x5 kernel of ones
kernel = np.ones((5,5),np.uint8)

# Erode the image
dilation = cv2.erode(image, kernel, iterations = 1)
```