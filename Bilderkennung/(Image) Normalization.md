Convert the scale of grey values from the range of 0 to 255 into the range between 0 and 1

norm_image = image / 255.0

```python
# normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32")/255
```

Normalization ensures that, as we go through a feedforward and then backpropagation step in training our CNN, that each image feature will fall within a similar range of values and not overly activate any particular layer in our network. During the feedfoward step, a network takes in an input image and multiplies each input pixel by some convolutional filter weights (and adds biases!), then it applies some activation and pooling functions. Without normalization, it's much more likely that the calculated gradients in the backpropagaton step will be quite large and cause our loss to increase instead of converge.