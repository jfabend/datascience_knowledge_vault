Oriented [[FAST Algorithm]] Rotated [[BRIEF Algorithm]]

The ORB algorithm searches the image for keypoints (like corners)
and then creates a binary feature vector for each of these keypoints. The set of feature vectors is called [[ORB Describtor]] and can be used for [[keypoint or feature matching]].
But ORB is rather suitable for objects with constistent features inside of the object like a face. For rather vaque objects like pedestrians it is better to use HOG

ORB is restistent against illuminations and rotation (u.a.) of the image ([[ORB Main Properties]]). Reason:

ORB uses an image pyramid of the original image ([[downscaling]]).
![[Pasted image 20210823161407.png|500]]

Then it calculates the keypoints on different scales on the different images and calculates the orientation (direction of intensity of these keypoints.
![[Pasted image 20210823161956.png]]

Afterwards, ORB used rotation-aware BRIEF (rBRIEF) to calculate feature vectors independently from orientation of the object


Before we implement the ORB Algorithm, we need to transform the image to [[RGB]] and by [[Grayscaling]]. After that, we can create ORB by setting the [[ORB Parameters]]. The first two arguments (nfeatures and scaleFactor) are probably the ones you are most likely to change. The other parameters can be safely left at their default values and you will get good result


```python
cv2.ORB_create(nfeatures = 500, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, firstLevel = 0, WTA_K = 2, scoreType = HARRIS_SCORE, patchSize = 31, fastThreshold = 20)
```
 
```python
# Import copy to make copies of the training image
import copy

# Set the default figure size
plt.rcParams['figure.figsize'] = [14.0, 7.0]

# Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
# the pyramid decimation ratio
orb = cv2.ORB_create(200, 2.0)

# Find the keypoints in the gray scale training image and compute their ORB descriptor.
# The None parameter is needed to indicate that we are not using a mask.
keypoints, descriptor = orb.detectAndCompute(training_gray, None)

# Create copies of the training image to draw our keypoints on
keyp_without_size = copy.copy(training_image)
keyp_with_size = copy.copy(training_image)

# Draw the keypoints without size or orientation on one copy of the training image 
cv2.drawKeypoints(training_image, keypoints, keyp_without_size, color = (0, 255, 0))

# Draw the keypoints with size and orientation on the other copy of the training image
cv2.drawKeypoints(training_image, keypoints, keyp_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with the keypoints without size or orientation
plt.subplot(121)
plt.title('Keypoints Without Size or Orientation')
plt.imshow(keyp_without_size)

# Display the image with the keypoints with size and orientation
plt.subplot(122)
plt.title('Keypoints With Size and Orientation')
plt.imshow(keyp_with_size)
plt.show()

# Print the number of keypoints detected
print("\nNumber of keypoints Detected: ", len(keypoints))
```
