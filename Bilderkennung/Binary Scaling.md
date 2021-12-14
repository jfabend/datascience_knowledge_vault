Wandelt z.B. eine Grauskalierung über einen Threshold um in Binary / Schwarz oder Weiß

```python
retval, binary_image = cv2.threshold(filtered_blurred, 50, 255, cv2.THRESH_BINARY)

plt.imshow(binary_image, cmap='gray')
```

Man kann das Binary Scaling auch invertiert machen, so dass quasi ein Negativ-Bild entsteht:

```python
retval, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
```