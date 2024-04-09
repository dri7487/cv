import skimage
import numpy as np
import cv2


orig = skimage.data.horse()
img= 255 - np.uint8(orig)*255
cv2.imshow("Horse", img)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

img2= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.drawContours(img2, contours,-1,(255,0,255),2)
cv2.imshow('Horse with contour',img2)

contour = contours[0]

m=cv2.moments(contour)
area= cv2.contourArea(contour)
cx, cy = m['m10']/m['m00'], m['m01']/m['m10'] 
perimeter = cv2.arcLength(contour, True)
roundness = (4.0*np.pi*area)/(perimeter*perimeter)
print("area:", area, f'midpoint: ({cx}, {cy})', 'perimerter:',perimeter,'roundness',roundness)

img3= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

contour_approx=cv2.approxPolyDP(contour,8,True)
cv2.drawContours(img3, [contour_approx], -1,(0,255,0),2 )

hull= cv2.convexHull(contour)
hull=hull.reshape(1, hull.shape[0],hull.shape[2])
cv2.drawContours(img3,hull,-1,(0,0,255),2)

cv2.imshow('Horse with line',img3)

cv2.waitKey()
cv2.destroyAllWindows()

