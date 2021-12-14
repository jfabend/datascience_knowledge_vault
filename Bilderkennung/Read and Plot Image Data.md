```python
import matplotlib.image as mpimg  # for reading in images
import matplotlib.pyplot as plt

image = mpimg.imread('images/waymo_car.jpg')

---- > Print out the image dimensions
print('Image dimensions:', image.shape)

----> Change from color to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image, cmap='gray')

----> Plot three images next to each other
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('R channel')
ax1.imshow(r, cmap='gray')
ax2.set_title('G channel')
ax2.imshow(g, cmap='gray')
ax3.set_title('B channel')
ax3.imshow(b, cmap='gray')

----> Change RGB Format
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)


IMAGE_LIST = helpers.load_dataset(image_dir_training)
selected_image = IMAGE_LIST[image_index][0]
selected_label = IMAGE_LIST[image_index][1]
- STANDARDIZED_LIST = helpers.standardize(IMAGE_LIST)
```

Plot each pixel with its grayscale value:
```python
# select an image by index
idx = 2
img = np.squeeze(images[idx])

# display the pixel values in that image
fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
```