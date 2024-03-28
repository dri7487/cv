import cv2 as cv
import sys

img=cv.imread('apples.jpg') #soccer.opg 라는 이미지 읽기

if img is None:
    sys.exit('파일이 존재하지 않습니다.') #soccer.jpg가 존재하지 않는다면 sys모듈 이용 종료

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY) #BGR 컬러를 명암 으로 변환
gray_small=cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5) #반으로 축소 = 크기 변환

    
#cv.imshow('image Display', img) # Image Display라는 창에 원본사진(img)
 
cv.imshow('image Display', gray) # Image Display라는 창에 명암사진(gray)
 #print(img[0,0,0], img[0,0,1], img[0,0,2]) #(0,0)의 B, G, R값 출력

cv.waitKey()
cv.destroyAllWindows() # 키보드 입력이 들어올떄 까지 켜놓기
# Image display 라는 창이 꺼지면 img의 type과 shape를 출력함.

print(type(img))
print(img.shape)
