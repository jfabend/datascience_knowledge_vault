

----> Define Color Thresholds
lower_blue = np.array([0,0,200]) 
upper_blue = np.array([250,250,255])

----> Define the masked area
mask = cv2.inRange(image_copy, lower_blue, upper_blue)

----> Visualize the mask
plt.imshow(mask, cmap='gray')

----> Mask the image to let the pizza show through
masked_image = np.copy(image_copy)
masked_image[mask != 0] = [0, 0, 0]

----> Display it!
plt.imshow(masked_image)

----> Lay two images / masks above each other
merged_image = image_one + background_image

----> Before you do that, make sure that the images have the same shape (by cropping it)
crop_background = background_image[0:514, 0:816]