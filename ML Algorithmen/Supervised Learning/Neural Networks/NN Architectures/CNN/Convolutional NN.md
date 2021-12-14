[An Intuitive Explanation of Convolutional Neural Networks – the data science blog (ujjwalkarn.me)](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

Before you implement a CNN by [[Code Example CNN PyTorch]] or [[Code Example 2 CNN Training PyTorch]] or [[Classify FashionMNIST, exercise]], you should perfom [[(Image) Normalization]]

The basic building blocks of any CNN are
1.  [[Convolution]] / [[Convolutional Layer]] / [[conv_layer_visualization]]
2.  [[Non-Linearity ReLu]]
3. [[Pooling Layer]] / [[pool_visualization]]

This Block of 1.-3. can be repeated several times. more layers = you can see more complex structures, but you should always consider the size and complexity of your training data (many layers may not be necessary for a simple task)

4. For the classification in the end (cat or dog), we need a [[Fully-Connected Layer]]
5. [[softmax]]
Additionally possible:
[[Dropout Layer]] between fully-connected layers

Simple CNN Architecure:
![[Pasted image 20210817110736.png]]

![[Pasted image 20210824193006.png]]

When you design and tune a CNN, you can check by [[NN Filter Activation Visualization PyTorch]] if the filters detect useful stuff or just random noise

Ein CNN lernt die optimale Konfiguration des [[Convolution]] Kernels. Man muss ihm hierfür aber die folgenden Parameter nennen:
- [[Depth]]
- [[Stride]]
- [[Zero-Padding]]

[[Famous CNN architectures]]