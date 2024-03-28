import cv2 as cv
import sys

img=cv.imread('soccer.jpg') #soccer.opg 라는 이미지 읽기

if img is None:
    sys.exit('파일이 존재하지 않습니다.') #soccer.jpg가 존재하지 않는다면 sys모듈 이용 종료

cv.rectangle(img,(290,780),(620,950),(0,0,255),2)
cv.putText(img,'mouse',(290,770),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) 

cv.imshow('Draw',img)

cv.waitKey()
cv.destroyAllWindows()