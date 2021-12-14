[[edge detection]] finds us all edges, but we need to determine only the object boundaries

![[Pasted image 20210820145051.png|600]]

Before the implementation of image contouring, we have to perform [[Binary Scaling]]

Dann kann man die Edges des weißen Objects bestimmen und der Kontour zuordnen, indem man
- alle Edge-Punkte über einen Threshold bestimmt ( ? nicht sicher ? )
- [[Canny Edge Detection]] durchführt

```python
# Find contours from thresholded, binary image
retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours on a copy of the original image
contours_image = np.copy(image)
contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3)

plt.imshow(contours_image)
```

From image contouring we can obtain contours and therefore [[Contour Features]]