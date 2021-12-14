image = mpimg.imread('images/wa_state_highway.jpg')
image.shape => (2000, 3000)

----> Find pixels with a certain value
import numpy as np
np.where(r == 0)
=> (array([ 977,  978,  979]), array([3365, 3365, 3365])

r[977, 3365]
=> 0