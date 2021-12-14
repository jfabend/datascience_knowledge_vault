[Channel](https://en.wikipedia.org/wiki/Channel_(digital_image)) is a conventional term used to refer to a certain component of an image. An image from a standard digital camera will have three channels – red, green and blue – you can imagine those as three 2d-matrices stacked over each other (one for each color), each having pixel values in the range 0 to 255.

The following color spaces contain different channels:
- [[RGB]]
- [[HSV]]
- [[HLS]]


![[Pasted image 20210817160223.png]]

----> Read in the image
image = mpimg.imread('images/wa_state_highway.jpg')

---->Isolate RGB channels
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]