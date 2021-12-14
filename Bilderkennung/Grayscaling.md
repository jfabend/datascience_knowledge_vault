A [grayscale](https://en.wikipedia.org/wiki/Grayscale "Grayscale") image has just one [[Channel]]. If we only consider grayscale images, we will have a single 2d matrix representing an image. The value of each pixel in the matrix will range from 0 to 255 – zero indicating black and 255 indicating white.

A colored picture has more channels and therefore a more complex matrix

```python
# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
```