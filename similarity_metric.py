import cv2 as cv
import numpy as np
import skimage.measure
from pathlib import Path
import pandas as pd
import re
import os


# Processing test images
directory = os.getcwd()
data_dir = Path(re.sub(r'[\\]','/',directory) + '/test_images')
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
			cropped_img = cv.resize(cropped_img, [64,64], interpolation = cv.INTER_CUBIC)
			# crop = cv.bilateralFilter(crop, 11, 75, 75)

			# cv.circle(crop,(i[0],i[1]),i[2],(0,255,0),2)
			# cv.circle(crop,(i[0],i[1]),2,(0,0,255),3)

	cv.imwrite(str(path).replace('test_images', 'cropped_test_images'), cropped_img)





# Support function

def MSE_metric(imgA, imgB):
	sum_diff=0.0
	# sum_A = 0.0
	# sum_B = 0.0
	for i in range(imgA.shape[0]):
		for j in range(imgA.shape[1]):
			sum_diff += (imgA[i,j,0] - imgB[i,j,0]) * (imgA[i,j,0] - imgB[i,j,0])
			# sum_A += imgA[i,j,0] * imgA[i,j,0]
			# sum_B += imgB[i,j,0] * imgB[i,j,0]

	for i in range(imgA.shape[0]):
		for j in range(imgA.shape[1]):
			sum_diff += (imgA[i,j,1] - imgB[i,j,1]) * (imgA[i,j,1] - imgB[i,j,1])
			# sum_A += imgA[i,j,1] * imgA[i,j,1]
			# sum_B += imgB[i,j,1] * imgB[i,j,1]

	for i in range(imgA.shape[0]):
		for j in range(imgA.shape[1]):
			sum_diff += (imgA[i,j,2] - imgB[i,j,2]) * (imgA[i,j,2] - imgB[i,j,2])
			# sum_A += imgA[i,j,2] * imgA[i,j,2]
			# sum_B += imgB[i,j,2] * imgB[i,j,2]

	return sum_diff / (3 * imgA.shape[0] * imgA.shape[1])
	# return sum_diff / (sum_A * sum_B)

def crop_circle(img):
  hh, ww = img.shape[:2]
  hc, wc = hh//2, ww//2
  radius = hh//2

  mask = np.zeros(img.shape[:2], dtype="uint8")
  # mask2 = np.zeros_like(img)
  mask = cv.circle(mask, (hc,wc), radius, (255,255), -1)
  cropped_img = cv.bitwise_and(img, img, mask=mask)
  return cropped_img



def blur(img):
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

data_dir = Path(re.sub(r'[\\]','/',directory) +'/templates')
paths_train = list((data_dir).glob('*.png'))[0:]

for path in paths_train:
	x = re.search(r'[\]][\w\.\_]+Original', str(path))
	hero_name = re.sub(r'[0-9]','',x.group()[1:-9]).lower()
	hero_names.append(hero_name)

	img = cv.imread(str(path))
	img = cv.resize(img, [64,64], interpolation = cv.INTER_AREA)
	hero_images.append(crop_circle(img))

# #print(len(hero_images))
# cv.imshow("template", hero_images[0])
# cv.waitKey(0)
# cv.destroyAllWindows()
# print(hero_images[0].shape)




# Load test data
with open(re.sub(r'[\\]','/',directory) +'/test.txt','r') as f:
	test_samples = f.readlines()

test_images = []
test_labels = []

test_data_path = re.sub(r'[\\]','/',directory) +'/cropped_test_images/'
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


print(len(hero_images))
print(len(hero_names))

# Predict test data
predict_labels = []
for img in test_images:
	predict_labels.append(predict(img,hero_images,hero_names))

acc = 0
for i in range(len(predict_labels)):
	if predict_labels[i] == test_labels[i]:
		acc += 1
print("Accuracy: ",acc,"/",len(test_labels))