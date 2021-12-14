FAST stands for Features from Accelerated Segments Test and finds the keypoints in an image.
By using it, we can define where in an image edges are, but not their direction or intensity.

It compares the brightness of the pixels in a given pixel area around pixel p with p:
- brighter than p
- darker than p
- similiar to p

If half of the given pixel area is either brighter or darker than p, p is a keypoint.
The pixels on the radius around p are the given pixel area:

![[Pasted image 20210823153710.png]]

An even faster variation of the FAST algorithm is to use only the 4 pixels on the very outside and declare p to keypoint, if two of them are either brighter or darker than p.

![[Pasted image 20210823154223.png]]