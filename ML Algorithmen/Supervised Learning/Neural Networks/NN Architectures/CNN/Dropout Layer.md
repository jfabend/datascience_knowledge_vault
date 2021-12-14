Avoids [[Overfitting]]

So wie man beim Fitness Workout nicht immer nur die starke Muskelgruppe trainieren will, so will man auch bei NN nicht nur die stark-gewichteten Netzbereiche trainieren.

Dropout layers essentially turn off certain nodes in a layer with some probability, p. This ensures that all nodes get an equal chance to try and classify different images during training, and it reduces the likelihood that only a few, heavily-weighted nodes will dominate the process

Dropout layers often come near the end of the network; placing them in between fully-connected layers for example can prevent any node in those layers from overly-dominating.
Kann man gut zwischen zwei [[Fully-Connected Layer]] bzw. [[Linear Layer]] einbauen:
```python
        # 20 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(20*5*5, 50)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(50, 10)
```

Dropout randomly turns off perceptrons (nodes) that make up the layers of our network, with some specified probability. It may seem counterintuitive to throw away a connection in our network, but as a network trains, some nodes can dominate others or end up making large mistakes, and dropout gives us a way to balance our network so that every node works equally towards the same goal, and if one makes a mistake, it won't dominate the behavior of our model. You can think of dropout as a technique that makes a network resilient; it makes all the nodes work well as a team by making sure no node is too weak or too strong. In fact it makes me think of the [Chaos Monkey](https://en.wikipedia.org/wiki/Chaos_Monkey) tool that is used to test for system/website failures.

I encourage you to look at the PyTorch dropout documentation, [here](http://pytorch.org/docs/stable/nn.html#dropout-layers), to see how to add these layers to a network.