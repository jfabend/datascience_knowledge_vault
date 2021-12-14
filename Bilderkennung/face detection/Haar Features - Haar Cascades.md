Before implementing Haar Cascades, we need to do  [[Grayscaling]]
```python
# load in cascade classifier
face_cascade = cv2.CascadeClassifier('parms.xml')

# run the detector on the grayscale image
faces = face_cascade.detectMultiScale(gray, 4, 6)
# Faces is a list of faces containing their coordinates in the image and their length&width
```

In the Params.xml, we need to define 

![[Pasted image 20210820101201.png|300]]

![[Pasted image 20210820101428.png|400]]
![[Pasted image 20210820101529.png|400]]