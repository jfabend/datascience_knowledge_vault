Can be used to detect vague objects like pedestrians due to
- [[Block Normalization]]
-  by filtering less interesting gradients by histogramm
-  by overlapping of the blocks the image block cells get normalized several times differently 

The result of HOG is the [[HOG Descriptor]]

[[Code Example HOG]]

![[Pasted image 20210824192118.png]]

![[Pasted image 20210824192300.png]]

![[Pasted image 20210824192424.png]]

The HOG algorithm works by creating histograms of the distribution of gradient orientations in an image and then normalizing them in a very special way. This special normalization is what makes HOG so effective at detecting the edges of objects even in cases where the contrast is very low. These normalized histograms are put together into a feature vector, known as the HOG descriptor, that can be used to train a machine learning algorithm

The HOG algorithm is implemented in a series of steps:

1.  Given the image of particular object, set a detection window (region of interest) that covers the entire object in the image (see Fig. 3).
    
2.  Calculate the magnitude and direction of the gradient for each individual pixel in the detection window.
    
3.  Divide the detection window into connected _cells_ of pixels, with all cells being of the same size (see Fig. 3). The size of the cells is a free parameter and it is usually chosen so as to match the scale of the features that want to be detected. For example, in a 64 x 128 pixel detection window, square cells 6 to 8 pixels wide are suitable for detecting human limbs.
    
4.  Create a Histogram for each cell, by first grouping the gradient directions of all pixels in each cell into a particular number of orientation (angular) bins; and then adding up the gradient magnitudes of the gradients in each angular bin (see Fig. 3). The number of bins in the histogram is a free parameter and it is usually set to 9 angular bins.
    
5.  Group adjacent cells into _blocks_ (see Fig. 3). The number of cells in each block is a free parameter and all blocks must be of the same size. The distance between each block (known as the stride) is a free parameter but it is usually set to half the block size, in which case you will get overlapping blocks (_see video below_). The HOG algorithm has been shown empirically to work better with overlapping blocks.
    
6.  Use the cells contained within each block to normalize the cell histograms in that block (see Fig. 3). If you have overlapping blocks this means that most cells will be normalized with respect to different blocks (_see video below_). Therefore, the same cell may have several different normalizations.
    
7.  Collect all the normalized histograms from all the blocks into a single feature vector called the HOG descriptor.
    
8.  Use the resulting HOG descriptors from many images of the same type of object to train a machine learning algorithm, such as an SVM, to detect those type of objects in images. For example, you could use the HOG descriptors from many images of pedestrians to train an SVM to detect pedestrians in images. The training is done with both positive a negative examples of the object you want detect in the image.
    
9.  Once the SVM has been trained, a sliding window approach is used to try to detect and locate objects in images. Detecting an object in the image entails finding the part of the image that looks similar to the HOG pattern learned by the SVM.

HOG creates histograms by adding the magnitude of the gradients in particular orientations in localized portions of the image called _cells_. By doing this we guarantee that stronger gradients will contribute more to the magnitude of their respective angular bin, while the effects of weak and randomly oriented gradients resulting from noise are minimized. In this manner the histograms tell us the dominant gradient orientation of each cell.

