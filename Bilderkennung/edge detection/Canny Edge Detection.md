````python
# use a gray scaled picture as input
lower = 50
upper = 100
edges = cv2.Canny(gray, lower, upper)
````

![[Pasted image 20210819154817.png|500]]

Canny Edge Detection uses Hysteresis to keep only the strong edges and the medium strong edges linked to the strong ones

![[Pasted image 20210819155054.png|500]]

The ratio between lower and upper boundary should be between 1 to 2 and 1 to 3