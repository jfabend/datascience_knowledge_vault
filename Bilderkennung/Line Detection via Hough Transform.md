For the implementation, apply [[Grayscaling]] and [[Canny Edge Detection]] to the image first. Then implement the line detection as follows:
````python
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 60
min_line_length = 50
max_line_gap = 5

line_image = np.copy(image) #creating an image copy to draw lines on

# Run Hough on the edge-detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)


# Iterate over the output "lines" and draw lines on the image copy
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
        
plt.imshow(line_image)
````

![[Pasted image 20210819160444.png|400]]

One line is represented by one point in the Hough Space:

![[Pasted image 20210820094120.png|400]]

One intersection point of several lines in Hough Space represent one line in the image
![[Pasted image 20210820094439.png|400]]
Many line points in Hough Space close to each other represent lines with a similiar mathematical function in the image space
![[Pasted image 20210820094740.png|400]]

=> When we look for intersection points in Hough Space then we find continous lines in image space.
Problem: A vertical line has an infinite function
=> That's why polar coordinates are used:
![[Pasted image 20210820095146.png|400]]

With polar coordinates, we also get four intersection points of the waves for four original lines in the image space:
![[Pasted image 20210820095320.png|200]] ![[Pasted image 20210820095446.png]]
