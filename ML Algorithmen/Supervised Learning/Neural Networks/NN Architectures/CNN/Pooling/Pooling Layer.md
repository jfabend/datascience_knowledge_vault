Reduziert die Feature Map / die Pixel-Matrix Dimensionen durch die Anwendung von Aggregationsfunktionen, zb [[Max Pooling]] , [[Global Average Pooling]], [[Pooling Layers PyTorch]]

Typischerweise eine Kernel-Size von 2 und ein Stride von 2

more layers = you can see more complex structures, but you should always consider the size and complexity of your training data (many layers may not be necessary for a simple task)

![[Pasted image 20210817120729.png|400]]

The function of Pooling is to progressively reduce the spatial size of the input representation [[4](http://cs231n.github.io/convolutional-networks/)]. In particular, pooling

-   makes the input representations (feature dimension) smaller and more manageable
-   reduces the number of parameters and computations in the network, therefore, controlling [[Overfitting]] (https://en.wikipedia.org/wiki/Overfitting) [[4](http://cs231n.github.io/convolutional-networks/)]
-   makes the network invariant to small transformations, distortions and translations in the input image (a small distortion in input will not change the output of Pooling – since we take the maximum / average value in a local neighborhood).
-   helps us arrive at an almost scale invariant representation of our image (the exact term is “equivariant”). This is very powerful since we can detect objects in an image no matter where they are located (read [[18](https://github.com/rasbt/python-machine-learning-book/blob/master/faq/difference-deep-and-normal-learning.md)] and [[19](https://www.quora.com/How-is-a-convolutional-neural-network-able-to-learn-invariant-features)] for details).