**Closing** is the reverse combination of [[Opening]]; it’s **dilation followed by erosion**, which is useful in _closing_ small holes or dark areas within an object.

Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.

![[Pasted image 20210820144308.png|300]]

```python
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```
