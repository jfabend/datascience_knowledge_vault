Short Explanation
With PyTorch you can
- define a class for a NN
- add the layers to different self variables / class attributes
- stick the layers together in the forward function

[[NN Layers PyTorch]]
[[Pooling Layers PyTorch]]
[[Loss Functions PyTorch]]
[[Optimizers PyTorch]]

[[Code Example CNN PyTorch]]
[[Code Example 2 CNN Training PyTorch]]

[[conv_layer_visualization]]
[[pool_visualization]]
[[NN Filter Activation Visualization PyTorch]]

[[load included datasets]]
[[dataloader]]

[[Flattening PyTorch]]

[[Save and load model]]


Long Explanation

PyTorch neural nets have their layers and feedforward behavior defined in a class. defining a network in a class means that you can instantiate multiple networks, dynamically change the structure of a model, and these class functions are called during training and testing.

PyTorch is also great for testing different model architectures, which is highly encouraged in this course! PyTorch networks are modular, which makes it easy to change a single layer in a network or modify the loss function and see the effect on training. If you'd like to see a review of PyTorch vs. TensorFlow, I recommend [this blog post](https://towardsdatascience.com/pytorch-vs-tensorflow-1-month-summary-35d138590f9).


