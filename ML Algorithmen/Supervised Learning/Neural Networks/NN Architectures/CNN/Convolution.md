Convolution with a Kernel:
![[Pasted image 20210817113813.png]]

Dabei werden die gerade vom Kernel bedeckten Pixel-Felder mit den Feldern der Kernel-Matrix mupltipliziert nach dem Schema
1x1=1
0x1=0
0x0=0

Die Summe des aktuellen Kernels wird dann in die Convolved Matrix (die der Größe des Kernels entspricht), geschrieben.

Für verschiedene Filter gibt es unterschiedliche Kernel-Konfigurationen, die das CNN allerdings lernen kann, z.b. [[high pass filters]] oder [[Sobel Filters]] für edge detection oder [[Gaussian Blur]] für Noise Reduction:
![[Pasted image 20210817115154.png]]

