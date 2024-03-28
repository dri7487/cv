import cv2 as cv
import sys

img=cv.imread('soccer.jpg') #soccer.opg 라는 이미지 읽기

if img is None:
    sys.exit('파일이 존재하지 않습니다.') #soccer.jpg가 존재하지 않는다면 sys모듈 이용 종료

def draw(event, x, y, flags, param):
    if event==cv.EVENT_LBUTTONDOWN:    
        cv.rectangle(img,(x,y),(x+200,y+200),(0,0,255),2)
    elif event==cv.EVENT_RBUTTDOWN:
        cv.rectangle(img,(x,y),(x+100,y+100),(255,0,0),2)
    cv.imshow('Drawing',img)
    
    
cv.namedWindow('Drawing')
cv.imshow('Drawing',img)

cv.setMouseCallback('Drawing',draw)
while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break

cv.imshow('Draw',img)

cv.waitKey()
cv.destroyAllWindows()