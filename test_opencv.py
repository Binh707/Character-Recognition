import cv2
import numpy as np

# # Load the image
# image = cv2.imread('vi.png')
# # cv2.imshow('Original Image', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Get the image dimensions
# height, width, _ = image.shape

# # Set the scale factor
# scale_factor = 1.2

# # Resize the image
# resized_image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
# cv2.imshow('Original Image', resized_image)
# print(resized_image.shape)

# # Calculate the difference in width and height
# delta_w = resized_image.shape[1] - width
# delta_h = resized_image.shape[0] - height

# # Add black pixels around the image
# # top, bottom = delta_h // 2, delta_h - (delta_h // 2)
# # left, right = delta_w // 2, delta_w - (delta_w // 2)

# top, bottom = 25, 25 
# left, right = 25, 25

# print(top)
# print(bottom)
# print(left)
# print(right)
# padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[20, 40, 30])
# print(padded_image.shape)

# # Display the original and padded images
# # cv2.imshow('Original Image', image)
# cv2.imshow('Padded Image', padded_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# # import cv2
# # import numpy as np

# # # read image 
# # img = cv2.imread('vi.png')

# # blur = cv2.GaussianBlur(img,(9,9),0)
# # blur = cv2.medianBlur(img,7)

# # cv2.imshow('image', img)
# # cv2.imshow('blur image', blur)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # hh, ww = img.shape[:2]
# # hh2 = hh // 2
# # ww2 = ww // 2

# # # define circles
# # radius1 = hh // 2
# # radius2 = hh // 2
# # xc = hh // 2
# # yc = ww // 2

# # # draw filled circles in white on black background as masks
# # # mask1 = np.zeros_like(img)
# # # mask1 = cv2.circle(mask1, (xc,yc), radius1, (255,255,255), -1)

# # mask2 = np.zeros(img.shape[:2], dtype="uint8")
# # # mask2 = np.zeros_like(img)
# # mask2 = cv2.circle(mask2, (xc,yc), radius2, (255,255), -1)

# # # subtract masks and make into single channel
# # # mask = cv2.subtract(mask2, mask1)
# # mask = mask2

# # # put mask into alpha channel of input
# # # result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
# # # result = img
# # # result[:, :, 2] = mask[:,:,0]

# # # save results
# # # cv2.imwrite('lena_mask1.png', mask1)
# # # cv2.imwrite('lena_mask2.png', mask2)
# # # cv2.imwrite('lena_masks.png', mask)
# # # cv2.imwrite('lena_circle_masks.png', result)

# # result = cv2.bitwise_and(img, img, mask=mask)

# # # cv2.imshow('image', img)
# # # # cv2.imshow('mask1', mask1)
# # # # cv2.imshow('mask2', mask2)
# # # cv2.imshow('mask', mask)
# # # cv2.imshow('masked image', result)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()


img = cv2.imread('IMG_E3983_auto_x2.jpg')
height, width, _ = img.shape
img = cv2.resize(img, (int(width // 4), int(height // 4)))
img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# img = cv2.filter2D(img, -1, kernel)

cv2.imshow('img1',img)
cv2.waitKey(0)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  
# https://github.com/Itseez/opencv/blob/master
# /data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Detects faces of different sizes in the input image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
for (x,y,w,h) in faces:
   print("Ho")
   # To draw a rectangle in a face 
   cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
   roi_gray = gray[y:y+h, x:x+w]
   roi_color = img[y:y+h, x:x+w]
  
   # # Detects eyes of different sizes in the input image
   # eyes = eye_cascade.detectMultiScale(roi_gray) 
  
   # #To draw a rectangle in eyes
   # for (ex,ey,ew,eh) in eyes:
   #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
  
    # Display an image in a window
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()






# hh,ww,cc = img.shape
# # img = cv2.GaussianBlur(img, (5,5), 0)
# # img = cv2.bilateralFilter(img, 11, 75, 75)
# # img = cv2.detailEnhance(img, sigma_s=50, sigma_r=0.15)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5,5), 0)
# # gray = cv2.bilateralFilter(gray, 11, 75, 75)

# # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# # gray = cv2.filter2D(gray, -1, kernel)
# # gray = cv2.filter2D(gray, -1, kernel)

# # gray = cv2.bilateralFilter(gray, 5, 75, 75)


# # Find Canny edges
# edged = cv2.Canny(gray, 50, 100)
# cv2.waitKey(0)
  
# # Finding Contours
# # Use a copy of the image e.g. edged.copy()
# # since findContours alters the image
# contours, hierarchy = cv2.findContours(edged, 
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
# cv2.imshow('Canny Edges After Contouring', edged)
# cv2.waitKey(0)
  
# print("Number of Contours found = " + str(len(contours)))

# for cnt in contours:
#    # x1,y1 = cnt[0][0]

#    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#    if len(approx) == 4 and cv2.contourArea(cnt):
#       x, y, w, h = cv2.boundingRect(cnt)
#       print(h," ",hh//2)

#       if h > hh//2:

#       	ratio = float(w)/h
#       	if ratio >= 0.9 and ratio <= 1.1:
#          	img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
#          	# cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#       	else:
#          	# cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#          	img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)

# # Draw all contours
# # -1 signifies drawing all contours
  
# cv2.imshow('Contours', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()