Damit kann man eine Art Spektralbild erstellen, welches angibt, welche Frequenzwerte es grundsätzlich im Bild gibt

![[Pasted image 20210819102702.png]]

Low frequencies are at the center of the frequency transform image. 

The transform images for these example show that the solid image has most low-frequency components into all directions. 

The stripes tranform image contains low-frequencies for the areas of white and black color and high frequencies for the edges in between those colors. The stripes transform image also tells us that there is one dominating direction for these frequencies; vertical stripes are represented by a horizontal line passing through the center of the frequency transform image.

![[Pasted image 20210818190926.png]]

![[Pasted image 20210818191124.png]]

![[Pasted image 20210819103216.png]]
Notice that this image has components of all frequencies. You can see a bright spot in the center of the transform image, which tells us that a large portion of the image is low-frequency; this makes sense since the body of the birds and background are solid colors. The transform image also tells us that there are **two** dominating directions for these frequencies; vertical edges (from the edges of birds) are represented by a horizontal line passing through the center of the frequency transform image, and horizontal edges (from the branch and tops of the birds' heads) are represented by a vertical line passing through the center.

```python
# perform a fast fourier transform and create a scaled, frequency transform image

def ft_image(norm_image):
    '''This function takes in a normalized, grayscale image
       and returns a frequency spectrum transform of that image. '''
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20*np.log(np.abs(fshift))
    
    return frequency_tx

# Apply the function to recieve the new image
f_stripes = ft_image(norm_stripes)
f_solid = ft_image(norm_solid)
```

For deeper understandment check out [[Fourier Trans of filters]]