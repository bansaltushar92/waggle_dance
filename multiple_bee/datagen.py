import sys
import cv2
import math
import os
import numpy as np
import pandas as pd
import random
import glob
import shutil
from sklearn import feature_extraction
from skimage.measure import compare_ssim



class Datagen(object):

	def __init__(self, video_id, video_path, annotations_path):
		self.cut_size = 40
		self.video_id = video_id
		self.video_path = video_path
		self.max_frames = 2000 ## Change to 20000
		self.window_path = '../windows/'
		self.transition_path = '../windows/transitions_renamed/'
		self.annotations_path = annotations_path


	def generate_negatives_df(self):
		
		self.column_names = ['frame_num', 'x','y']
		self.negative_df = pd.DataFrame(columns=self.column_names)
		cap = cv2.VideoCapture(self.video_path)
		frame_num = 0

		for _ in range((self.max_frames/10)):
			ret, img = cap.read()
			if ret:
				img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				if frame_num % 10 == 0:
					for _ in range(0,20):
						rand_x = np.random.randint(low=self.cut_size+1, high=img_gray.shape[1]-self.cut_size, size=1)[0]
						rand_y = np.random.randint(low=self.cut_size+1, high=img_gray.shape[0]-self.cut_size, size=1)[0]
						self.negative_df = self.negative_df.append({'frame_num': frame_num, 'x': rand_x, 'y': rand_y}, ignore_index=True)
				frame_num+=1


	def generate_negative_windows(self):
		
		cap = cv2.VideoCapture(self.video_path)
		frame_num = 0

		for _ in range(self.max_frames):
			ret, img = cap.read()
			if ret:
				img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				for trail in range(5,10,5):
					if self.negative_df[(self.negative_df['frame_num'] == frame_num+trail)].shape[0] > 0:
						for idx, row in self.negative_df[(self.negative_df['frame_num'] == frame_num+trail)].iterrows():
							img_square = img_gray[row['y']-self.cut_size:row['y']+self.cut_size, row['x']-self.cut_size:row['x']+self.cut_size]
							path = self.window_path + 'final_negatives_transitions/'
							if not os.path.exists(path):
								os.makedirs(path)
							cv2.imwrite(os.path.join(path,  str(self.video_id) + '_' + str(frame_num+trail) + '_'+ str(row['x']) + '_'+ str(row['y']) + '_t'+ str(trail) + '.jpg'), img_square)
				for lead in range(5,10,5):
					if self.negative_df[(self.negative_df['frame_num'] == frame_num-lead)].shape[0] > 0:
						for idx, row in self.negative_df[(self.negative_df['frame_num'] == frame_num-lead)].iterrows():
							img_square = img_gray[row['y']-self.cut_size:row['y']+self.cut_size, row['x']-self.cut_size:row['x']+self.cut_size]
							path = self.window_path + 'final_negatives_transitions/'
							if not os.path.exists(path):
								os.makedirs(path)
							cv2.imwrite(os.path.join(path , str(self.video_id) + '_' + str(frame_num-lead) + '_'+ str(row['x']) + '_'+ str(row['y']) + '_l'+ str(lead) + '.jpg'), img_square)
				if self.negative_df[(self.negative_df['frame_num'] == frame_num)].shape[0] > 0:
					for idx, row in self.negative_df[(self.negative_df['frame_num'] == frame_num)].iterrows():
						img_square = img_gray[row['y']-self.cut_size:row['y']+self.cut_size, row['x']-self.cut_size:row['x']+self.cut_size]
						path = self.window_path + 'final_negatives/'
						if not os.path.exists(path):
								os.makedirs(path)
						cv2.imwrite(os.path.join(path , str(self.video_id) + '_' + str(frame_num) + '_'+ str(row['x']) + '_'+ str(row['y']) + '.jpg'), img_square)
				frame_num+=1

	def generate_positives_df(self):
		
		ann_df = pd.read_csv(self.annotations_path)
		self.positive_df = ann_df.loc[((ann_df['behavior'] == 'wS') | (ann_df['behavior'] == 'wE')), ['frame', 'x', 'y', 'behavior']]
		self.positive_df.columns = ['frame_num', 'x', 'y', 'behavior']
		self.positive_df['x'] = self.positive_df['x']+1920/2
		self.positive_df['y'] = 1080/2-self.positive_df['y']
		self.positive_df.x = self.positive_df.x.astype(int)
		self.positive_df.y = self.positive_df.y.astype(int)

	def generate_positive_windows(self):

		cap = cv2.VideoCapture(self.video_path)
		frame_num = 0

		for _ in range(self.max_frames):
			ret, img = cap.read()
			if ret:
				img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				for trail in range(5,10,5):
					if self.positive_df[(self.positive_df['frame_num'] == frame_num+trail)].shape[0] > 0:
						for idx, row in self.positive_df[(self.positive_df['frame_num'] == frame_num+trail)].iterrows():
							img_square = img_gray[row['y']-self.cut_size:row['y']+self.cut_size, row['x']-self.cut_size:row['x']+self.cut_size]
							path = self.window_path + 'final_positives_transitions/'
							if not os.path.exists(path):
								os.makedirs(path)
							cv2.imwrite(os.path.join(path , str(self.video_id) + '_' + str(frame_num+trail) + '_'+ str(row['x']) + '_'+ str(row['y']) + '_t'+ str(trail) + '.jpg'), img_square)
				for lead in range(5,10,5):
					if self.positive_df[(self.positive_df['frame_num'] == frame_num-lead)].shape[0] > 0:
						for idx, row in self.positive_df[(self.positive_df['frame_num'] == frame_num-lead)].iterrows():
							img_square = img_gray[row['y']-self.cut_size:row['y']+self.cut_size, row['x']-self.cut_size:row['x']+self.cut_size]
							path = self.window_path + 'final_positives_transitions/'
							if not os.path.exists(path):
								os.makedirs(path)
							cv2.imwrite(os.path.join(path , str(self.video_id) + '_' + str(frame_num-lead) + '_'+ str(row['x']) + '_'+ str(row['y']) + '_l'+ str(lead) + '.jpg'), img_square)
				if self.positive_df[(self.positive_df['frame_num'] == frame_num)].shape[0] > 0:
					for idx, row in self.positive_df[(self.positive_df['frame_num'] == frame_num)].iterrows():
						img_square = img_gray[row['y']-self.cut_size:row['y']+self.cut_size, row['x']-self.cut_size:row['x']+self.cut_size]
						path = self.window_path + 'final_positives/'
						if not os.path.exists(path):
								os.makedirs(path)
						cv2.imwrite(os.path.join(path , str(self.video_id) + '_' + str(frame_num) + '_'+ str(row['x']) + '_'+ str(row['y']) + '.jpg'), img_square)
				frame_num+=1


	# def transition_windows(self):

	# 	cap = cv2.VideoCapture(self.video_path)
	# 	frame_num = 0

	# 	for _ in range(self.max_frames):
	# 		ret, img = cap.read()
	# 		if ret:
	# 			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				
	# 			if self.positive_df[(self.positive_df['frame_num'] == frame_num + 5) & (self.positive_df['behavior'] == 'wE')].shape[0] > 0:
	# 				for idx, row in self.positive_df[(self.positive_df['frame_num'] == frame_num + 5) & (self.positive_df['behavior'] == 'wE')].iterrows():
	# 					img_square = img_gray[row['y']-self.cut_size:row['y']+self.cut_size, row['x']-self.cut_size:row['x']+self.cut_size]
	# 					path = self.window_path + 'final_positives_diff_5/'
	# 					cv2.imwrite(os.path.join(path , str(frame_num+5) + '_'+ str(row['x']) + '_'+ str(row['y']) + '.jpg'), img_square)
				
	# 			if self.positive_df[(self.positive_df['frame_num'] == frame_num - 5) & (self.positive_df['behavior'] == 'wS')].shape[0] > 0:
	# 				for idx, row in self.positive_df[(self.positive_df['frame_num'] == frame_num - 5) & (self.positive_df['behavior'] == 'wS')].iterrows():
	# 					img_square = img_gray[row['y']-self.cut_size:row['y']+self.cut_size, row['x']-self.cut_size:row['x']+self.cut_size]
	# 					path = self.window_path + 'final_positives_diff_5/'
	# 					cv2.imwrite(os.path.join(path , str(frame_num-5) + '_'+ str(row['x']) + '_'+ str(row['y']) + '.jpg'), img_square)

	# 			frame_num+=1


	def transition_positives(self):

		if not os.path.exists(self.transition_path):
			os.makedirs(self.transition_path)

		img_list = glob.glob(self.window_path + 'final_positives_transitions/*')

		for img_name in img_list:
			img = cv2.imread(img_name)
			img_name_list = img_name.split('/')[-1].split('.')[0].split('_')
			if img_name_list[4][0:1] == 't':
				cv2.imwrite(os.path.join(self.transition_path , img_name_list[0] + '_' + str(int(img_name_list[1])-int(img_name_list[4][1:])) + '_'+ img_name_list[2] + '_'+ img_name_list[3] + '.jpg'), img)
			elif img_name_list[4][0:1] == 'l':
				cv2.imwrite(os.path.join(self.transition_path , img_name_list[0] + '_' + str(int(img_name_list[1])+int(img_name_list[4][1:])) + '_'+ img_name_list[2] + '_'+ img_name_list[3] + '.jpg'), img)
			else:
				print 'window not found: ',  img_name_list, ' : ' , img_name_list[4][0:1]


	def transition_negatives(self):

		if not os.path.exists(self.transition_path):
			os.makedirs(self.transition_path)
		
		img_list = glob.glob(self.window_path + 'final_negatives_transitions/*')
		
		for img_name in img_list:
			img = cv2.imread(img_name)
			img_name_list = img_name.split('/')[-1].split('.')[0].split('_')
			if img_name_list[4][0:1] == 't':
				cv2.imwrite(os.path.join(self.transition_path , img_name_list[0] + '_' + str(int(img_name_list[1])-int(img_name_list[4][1:])) + '_'+ img_name_list[2] + '_'+ img_name_list[3] + '.jpg'), img)
			elif img_name_list[4][0:1] == 'l':
				cv2.imwrite(os.path.join(self.transition_path , img_name_list[0] + '_' + str(int(img_name_list[1])+int(img_name_list[4][1:])) + '_'+ img_name_list[2] + '_'+ img_name_list[3] + '.jpg'), img)
			else:
				print 'window not found: ', img_name_list, ' : ' , img_name_list[4][0:1]



	def get_dataframe(self):

		diff_data = []
		labels = np.array([])
		img_list = glob.glob(self.window_path + 'final_positives_transitions/*')
		transition_list = glob.glob(self.transition_path + '*')

		for img_name in img_list:
			img = cv2.imread(img_name)[:,:,0]
			img_name_list = img_name.split('/')[-1].split('.')[0].split('_')
			img_name_list[1] = str(int(img_name_list[1])-5)
			img_prev_name = '_'.join(img_name_list[:-1]) + '.jpg'
			img_prev_name = self.transition_path + img_prev_name
			img_prev = cv2.imread(img_prev_name)[:,:,0]
			(score, diff) = compare_ssim(img, img_prev, full=True)
			diff = (diff * 255).astype("uint8")
			diff_data.append(np.concatenate((diff.flatten(), img.flatten()), axis=0))
			labels  = np.append(labels, [1])

		img_list = glob.glob(self.window_path + 'final_negatives_transitions/*')

		for img_name in img_list:
			img = cv2.imread(img_name)[:,:,0]
			img_name_list = img_name.split('/')[-1].split('.')[0].split('_')
			img_name_list[1] = str(int(img_name_list[1])-5)
			img_prev_name = '_'.join(img_name_list[:-1]) + '.jpg'
			img_prev_name = self.transition_path + img_prev_name
			if img_prev_name in transition_list:
				img_prev = cv2.imread(img_prev_name)[:,:,0]
				(score, diff) = compare_ssim(img, img_prev, full=True)
				diff = (diff * 255).astype("uint8")
				diff_data.append(np.concatenate((diff.flatten(), img.flatten()), axis=0))
				labels  = np.append(labels, [0])

		return np.array(diff_data), labels


	def compute_hist(self, img):
		return cv2.calcHist([img],[0],None,[256],[0,256]).flatten()

	def compute_hist_multiple(self, img):
		return np.concatenate((cv2.calcHist([img[:6400]],[0],None,[256],[0,256]).flatten(), \
			cv2.calcHist([img[6400:]],[0],None,[256],[0,256]).flatten()), axis=0)

	def convert_hist_data(self, diff_data):
		
		hist_data = []
		for row in diff_data:
			hist_data.append(self.compute_hist_multiple(row))

		return hist_data

	def generate_data(self):

		if not os.path.exists(self.window_path):
			os.makedirs(self.window_path)

		self.generate_negatives_df()
		self.generate_negative_windows()
		self.generate_positives_df()
		self.generate_positive_windows()
		self.transition_positives()
		self.transition_negatives()
		diff_data, labels = self.get_dataframe()
		hist_data = self.convert_hist_data(diff_data)

		if os.path.exists(self.window_path):
			shutil.rmtree(self.window_path)

		return hist_data, labels









