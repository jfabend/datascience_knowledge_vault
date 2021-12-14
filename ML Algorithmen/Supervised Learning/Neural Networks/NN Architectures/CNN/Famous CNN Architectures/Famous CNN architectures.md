-   **LeNet (1990s):** Already covered in this article.

-   **1990s to 2012:** In the years from late 1990s to early 2010s convolutional neural network were in incubation. As more and more data and computing power became available, tasks that convolutional neural networks could tackle became more and more interesting.

-   **AlexNet (2012) –** In 2012, Alex Krizhevsky (and others) released [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) which was a deeper and much wider version of the LeNet and won by a large margin the difficult ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. It was a significant breakthrough with respect to the previous approaches and the current widespread application of CNNs can be attributed to this work. [A Walk-through of AlexNet. AlexNet famously won the 2012 ImageNet… | by Hao Gao | Medium](https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637 [ImageNet Classification with Deep Convolutional Neural Networks (nips.cc)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

-   **ZF Net (2013) –** The ILSVRC 2013 winner was a Convolutional Network from Matthew Zeiler and Rob Fergus. It became known as the [ZFNet](http://arxiv.org/abs/1311.2901) (short for Zeiler & Fergus Net). It was an improvement on AlexNet by tweaking the architecture hyperparameters.

-   **GoogLeNet (2014) –** The ILSVRC 2014 winner was a Convolutional Network from [Szegedy et al.](http://arxiv.org/abs/1409.4842) from Google. Its main contribution was the development of an _Inception Module_ that dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M).

-   **VGGNet (2014) –** The runner-up in ILSVRC 2014 was the network that became known as the [VGGNet](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). Its main contribution was in showing that the depth of the network (number of layers) is a critical component for good performance.

[[ResNet - Residual Network]]

-   **DenseNet (August 2016) –** Recently published by Gao Huang (and others), the [Densely Connected Convolutional Network](http://arxiv.org/abs/1608.06993) has each layer directly connected to every other layer in a feed-forward fashion. The DenseNet has been shown to obtain significant improvements over previous state-of-the-art architectures on five highly competitive object recognition benchmark tasks. Check out the Torch implementation [here](https://github.com/liuzhuang13/DenseNet).


[[WaveNet]]