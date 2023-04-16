import cv2 as cv
import numpy as np
import skimage.measure
from pathlib import Path
import pandas as pd
import re

data_dir = Path('D:/Project/CV/nan_junior/AI-Engineer-Test/test_data.tar/test_data/test_images')
paths_train = list((data_dir).glob('*.jpg'))[0:]

for path in paths_train:

	img = cv.imread(str(path))
	hh, ww = img.shape[0:2]
	crop = img[0 : img.shape[0], 0 : ww//2]

	# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	# crop = cv.filter2D(crop, -1, kernel)
	# crop = cv.detailEnhance(crop, sigma_s=10, sigma_r=0.15)

	# crop = cv.GaussianBlur(crop, (7,7), 1.5)
	crop = cv.bilateralFilter(crop, 11, 75, 75)
	# crop = cv.medianBlur(crop, 5)
	# crop = cv.fastNlMeansDenoisingColored(crop,None,20,20,7,21)

	# crop = cv.detailEnhance(crop, sigma_s=10, sigma_r=0.3)

	

	gray_img = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
	circles_img = cv.HoughCircles(gray_img,  cv.HOUGH_GRADIENT,1.0, ww,
                              param1=50, param2=10, minRadius=hh//5, maxRadius=hh//2+5)

	if circles_img is not None:
		circles_img = np.uint16(np.around(circles_img))
		for i in circles_img[0,:]:

			# Crop square of character's avatar
			r = i[2]
			if i[0] <= r:
				left = 0
			else:
				left = i[0]-r

			right = i[0] + r

			if i[1] <= r:
				top = 0
			else:
				top = i[1]-r

			bottom = i[1] + r
			crop = img[top:bottom, left:right]
			crop = cv.detailEnhance(crop, sigma_s=40, sigma_r=0.7)
			# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
			# crop = cv.filter2D(crop, -1, kernel)

			# Crop circle and convert to size 128x128
			square_hh, square_ww = crop.shape[:2]
			square_hc, square_wc = square_hh//2, square_hh//2
			radius = square_hh//2

			mask = np.zeros(crop.shape[:2], dtype="uint8")
			mask = cv.circle(mask, (square_hc, square_wc), radius, (255,255), -1)
			cropped_img = cv.bitwise_and(crop, crop, mask=mask)
			cropped_img = cv.resize(cropped_img, [32,32], interpolation = cv.INTER_AREA)
			# crop = cv.bilateralFilter(crop, 11, 75, 75)

			# cv.circle(crop,(i[0],i[1]),i[2],(0,255,0),2)
			# cv.circle(crop,(i[0],i[1]),2,(0,0,255),3)

	cv.imwrite(str(path).replace('test_images', 'preprocessing_images'), cropped_img)

# img = cv.imread("test_data.tar/test_data/test_images/Akali_9Aa4KRvaLFA_round3_Fizz_05-19-2021.mp4_82_1.jpg")

# # print(img.shape)
# mid = int(img.shape[1]/2)

# crop = img[0:img.shape[0], 0:mid]
# # crop = img
# cv.imshow('crop_1', crop)

# crop = cv.detailEnhance(crop, sigma_s=4, sigma_r=0.15)
# cv.imshow('crop_2', crop)

# # crop = cv.detailEnhance(crop, sigma_s=10, sigma_r=0.15)
# # cv.imshow('crop_3', crop)

# cv.waitKey(0)
# cv.destroyAllWindows()
# # print(crop.shape)

# # crop = cv.resize(crop, [128,128], interpolation = cv.INTER_AREA)
# # # crop = crop / 255.0


# # crop = cv.detailEnhance(crop, sigma_s=10, sigma_r=0.5)

# # # print(crop.shape)

# # cv.imshow("cropped", crop)
# # cv.waitKey(0)
# # cv.destroyAllWindows()

# # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# # # Apply the sharpening kernel to the image using filter2D
# # crop = cv.filter2D(crop, -1, kernel)

 

# # cv.imshow("cropped", crop)
# # cv.waitKey(0)
# # cv.destroyAllWindows()

# # g_crop = np.expand_dims(skimage.measure.block_reduce(crop[:,:,0], (4,4), np.max), axis=2)
# # r_crop = np.expand_dims(skimage.measure.block_reduce(crop[:,:,1], (4,4), np.max), axis=2)
# # b_crop = np.expand_dims(skimage.measure.block_reduce(crop[:,:,2], (4,4), np.max), axis=2)

# # new_crop = np.concatenate([g_crop,r_crop,b_crop], axis=2)
# # print(new_crop.shape) 

# # cv.imwrite('test.jpg', new_crop)
# # new_crop = cv.resize(new_crop, [128,128], interpolation = cv.INTER_AREA)
# # cv.imshow("cropped", new_crop)
# # cv.waitKey(0)
# # cv.destroyAllWindows()



# gray_img = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
# # circles_img = cv.HoughCircles(gray_img,cv.HOUGH_GRADIENT,0.7,50,
# #                             param1=50,param2=20,minRadius=20,maxRadius=30)

# hh,ww = gray_img.shape
# # ww = ww // 2
# hh = hh // 2

# circles_img = cv.HoughCircles(gray_img,cv.HOUGH_GRADIENT,0.7,ww,
#                              param1=50,param2=40,minRadius=0,maxRadius=hh)

# if circles_img is not None:
# 	circles_img = np.uint16(np.around(circles_img))
# 	for i in circles_img[0,:]:
# 		cv.circle(crop,(i[0],i[1]),i[2],(0,255,0),2)
# 		cv.circle(crop,(i[0],i[1]),2,(0,0,255),3)

# 	cv.imshow('Detected Circles',crop)
# 	cv.waitKey(0)
# 	cv.destroyAllWindows()


# Support function

def MSE_metric(imgA, imgB):
	sum_diff=0.0
	sum_A = 0.0
	sum_B = 0.0
	for i in range(imgA.shape[0]):
		for j in range(imgA.shape[1]):
			sum_diff += (imgA[i,j,0] - imgB[i,j,0]) * (imgA[i,j,0] - imgB[i,j,0])
			sum_A += imgA[i,j,0] * imgA[i,j,0]
			sum_B += imgB[i,j,0] * imgB[i,j,0]

	for i in range(imgA.shape[0]):
		for j in range(imgA.shape[1]):
			sum_diff += (imgA[i,j,1] - imgB[i,j,1]) * (imgA[i,j,1] - imgB[i,j,1])
			sum_A += imgA[i,j,1] * imgA[i,j,1]
			sum_B += imgB[i,j,1] * imgB[i,j,1]

	for i in range(imgA.shape[0]):
		for j in range(imgA.shape[1]):
			sum_diff += (imgA[i,j,2] - imgB[i,j,2]) * (imgA[i,j,2] - imgB[i,j,2])
			sum_A += imgA[i,j,2] * imgA[i,j,2]
			sum_B += imgB[i,j,2] * imgB[i,j,2]

	# return sum_diff / (3 * imgA.shape[0] * imgA.shape[1])
	return sum_diff / (sum_A * sum_B)

def aug_crop_circle(img):
  hh, ww = img.shape[:2]
  hc, wc = hh//2, ww//2
  radius = hh//2

  mask = np.zeros(img.shape[:2], dtype="uint8")
  # mask2 = np.zeros_like(img)
  mask = cv.circle(mask, (hc,wc), radius, (255,255), -1)
  cropped_img = cv.bitwise_and(img, img, mask=mask)
  return cropped_img



def aug_bulr(img):
  # blured_img = cv.medianBlur(img,7)
  blured_img = cv.GaussianBlur(img, (7,7), 0)
  return blured_img

def predict(test_image, hero_images, hero_names):
	mse_list = []
	for img in hero_images: 
		mse_list.append(MSE_metric(img, test_image))

	hero_predict = mse_list.index(min(mse_list))
	return hero_names[hero_predict]




# Load standrad character's name and template
hero_names = []
hero_images = []

data_dir = Path('D:/Project/CV/nan_junior/AI-Engineer-Test/standrad_image')
paths_train = list((data_dir).glob('*.png'))[0:]

for path in paths_train:
	x = re.search(r'[\]][\w\.\_]+Original', str(path))
	hero_name = re.sub(r'[0-9]','',x.group()[1:-9]).lower()
	hero_names.append(hero_name)

	img = cv.imread(str(path))
	img = cv.resize(img, [32,32], interpolation = cv.INTER_AREA)
	hero_images.append(aug_crop_circle(img))

# #print(len(hero_images))
# cv.imshow("template", hero_images[0])
# cv.waitKey(0)
# cv.destroyAllWindows()
# print(hero_images[0].shape)



# Load test data
with open('D:/Project/CV/nan_junior/AI-Engineer-Test/test_data.tar/test_data/test.txt','r') as f:
	test_samples = f.readlines()
# print(test_samples)

test_images = []
test_labels = []

test_data_path = 'D:/Project/CV/nan_junior/AI-Engineer-Test/test_data.tar/test_data/preprocessing_images/'
for sample in test_samples:
	file_name, label = sample.split('\t')
	label = re.sub(r'[0-9]','',label).lower()
	test_labels.append(label.strip('\n'))

	img = cv.imread(test_data_path + file_name)
	test_images.append(img)

# cv.imshow('test', test_images[25])
# cv.waitKey(0)
# cv.destroyAllWindows()

# print(test_labels[25])

# print(predict(test_images[25],hero_images,hero_names))

# Predict test data
predict_labels = []
for img in test_images:
	predict_labels.append(predict(img,hero_images,hero_names))

acc = 0
for i in range(len(predict_labels)):
	if predict_labels[i] == test_labels[i]:
		acc += 1
print("Accuracy: ",acc,"/",len(test_labels))
