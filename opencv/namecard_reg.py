import cv2
import numpy as np
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
'''
img = cv2.imread("./swordman.jpg")
#img = cv2.imread("./man.jpg")
#img = cv2.imread("./000260.png")
h, w = img.shape[:2]
blured = cv2.medianBlur(img, 3) # medianBlur效果比blur好 blured = cv2.blur(img, (2,2))

mask = np.zeros((h+2, w+2), np.uint8) #掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
#进行泛洪填充
cv2.floodFill(blured, mask, (w-1,h-1), (255,255,255), (2,2,2),(3,3,3),8)
#cv2.imshow("floodfill", blured) 

gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
ret, binary = cv2.threshold(closed,100,255,cv2.THRESH_BINARY) 
#ret, binary = cv2.threshold(closed,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY) 
#binary = cv2.Canny(closed,15,100)
#cv2.imshow("img", binary)
contours= cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contour_outer = contours[0]
#创建白色幕布
#temp = np.ones(binary.shape,np.uint8)*255
#画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
#cv2.drawContours(temp,contour_outer,-1,(0,0,255),3)
cv2.drawContours(img,contour_outer,-1,(0,0,255),3) 
cv2.imshow("img", img)
cv2.waitKey(0)
#print (type(contours))
#print (type(contours[0]))
#print (len(contours))
#print (len(contours[0]))