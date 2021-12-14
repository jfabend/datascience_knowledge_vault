![[Pasted image 20210819154106.png|600]]

![[Pasted image 20210824195354.png]]

For [[RGB]] images with depth=3 we also need a convolutional kernel with the depth of 3

![[Pasted image 20210824195817.png]]

Typischerweise liegt die Kernel-Size bei 3 oder 5. Je größer die Bilder, desto größer die Kernel-Size

Die Anzahl der Output-Kanäle übersteigt die Anzahl der Input-Kanäle, z.B. von 1 auf 10 oder von 10 auf 20.

more layers = you can see more complex structures, but you should always consider the size and complexity of your training data (many layers may not be necessary for a simple task)

[[Conv Layer Filter Progress Viz]]