import cv2

img = cv2.imread(imgfile)

gray_img = cv2.cvtColor(img.cv2.COLOR_BGR2RGB)

inverted_img = 255-gray_img

blur = cv2.GaussianBlur(inverted_img,(23,23),0)
intervted_blur = 255- blur
pencil_sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)

cv2.imshow('original image', img)
cv2.imshow('pencil sketch' pencil_sketch)

cv2.waitKey(0)
