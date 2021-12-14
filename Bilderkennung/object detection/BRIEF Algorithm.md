The BRIEF algorithm takes the keypoints found by the [[FAST Algorithm]] and calculates feature vectors for each keypoint.

Binary
Robust
Independent
Elementary
Features

![[Pasted image 20210823155258.png|400]]

Steps:
- [[Gaussian Blur]] to remove high frequency noise
- random selection of two pixels in the neighbourhood around the keypoint and comparison of their brightness => 1 if first pixel is brighter, 0 if not