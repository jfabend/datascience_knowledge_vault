As mentioned, above, these operations are often _combined_ for desired results! One such combination is called **opening**, which is **erosion followed by dilation**. This is useful in noise reduction in which erosion first gets rid of noise (and shrinks the object) then dilation enlarges the object again, but the noise will have disappeared from the previous erosion! It is the reverse combination of [[Closing]].

![[Pasted image 20210820144007.png|300]]

To implement this in OpenCV, we use the function morphologyEx with our original image, the operation we want to perform, and our kernel passed in.

```pythpn
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
```