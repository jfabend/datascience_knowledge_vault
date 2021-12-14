- if there is no shadow, you can use [[RGB]] color space
	- As lower boundary for the mask set the other colors to 0 and the desired one to a medium value (=> dark green eg., almost black)
	- As upper boundary set all color values pretty high but the desired one must be the highest (=> very light green eg., almost white)
	- Then generate the [[Masks - Area Selection]]
- If there are different light conditions in the image, use the [[HSV]] color space
	- plot the the Hue channel (index 0) and check out which degree area describes your object (color) the best for the lower mask boundary
		- object is almost black => use a low hue range (0-30 ?)
		- object is grey => use a medium hue range (60-100 ?)
		- object is almost white => use a high hue range (150-180)
	- use (180, 255, 255) for the upper mask boundary