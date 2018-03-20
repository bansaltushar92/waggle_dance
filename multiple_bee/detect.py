from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from skimage.measure import compare_ssim
import numpy as np
import pandas as pd
import argparse
import pickle
import cv2


class Main(object):

	def __init__(self, video_path, model_path, result_path):
		self.video_path = video_path
		self.result_path = result_path
		self.model_path = model_path
		self.cut_size = 40
		self.max_frames = 20 ## Change to 20000

	def compute_hist_multiple(self, img):
		return np.concatenate((cv2.calcHist([img[:6400]],[0],None,[256],[0,256]).flatten(),
						  cv2.calcHist([img[6400:]],[0],None,[256],[0,256]).flatten()), axis=0)

	def detect(self):

		file = open(self.model_path,'rb')
		(clf_xgb, clf_rf) = pickle.load(file)
		file.close()

		cap = cv2.VideoCapture(self.video_path)
		last_five_frames = []
		this_w_pred = {}
		prev_w_pred = {}
		test_data_list = []

		for frame_num in range(self.max_frames):
			ret, img = cap.read()
			if ret:
				img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				if frame_num % 10 == 0 and frame_num>0:
					for cnt_x in range(self.cut_size, img_gray.shape[1]-self.cut_size, self.cut_size/2):
						for cnt_y in range(self.cut_size, img_gray.shape[0]-self.cut_size, self.cut_size/2):
							img_square = img_gray[cnt_y-self.cut_size:cnt_y+self.cut_size, cnt_x-self.cut_size:cnt_x+self.cut_size]
							img_prev_square = last_five_frames[0][cnt_y-self.cut_size:cnt_y+self.cut_size, cnt_x-self.cut_size:cnt_x+self.cut_size]
							(score, diff) = compare_ssim(img_square, img_prev_square, full=True)
							diff = (diff * 255).astype("uint8")
							test_input = self.compute_hist_multiple(np.concatenate((diff.flatten(), img_square.flatten()), axis=0)).reshape(1,512)
							test_data = np.concatenate((np.array([[frame_num, cnt_x, cnt_y]]),test_input), 1)
							test_data_list.append(test_data[0])
					last_five_frames = last_five_frames[-5:]
					prev_w_pred = this_w_pred
				last_five_frames.append(img_gray)

		columns = ['frame_num','x', 'y'] + range(512)
		test_data_df = pd.DataFrame(test_data_list, columns=columns)

		train_cols = range(512)					
		preds = clf_xgb.predict(test_data_df[train_cols].as_matrix())*clf_rf.predict(test_data_df[train_cols].as_matrix())
		test_data_df['preds'] = preds
		test_data_df.to_csv('result.csv')

		cap = cv2.VideoCapture(self.video_path)
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		video = cv2.VideoWriter(self.result_path,fourcc,30.0,(960,540))
		for frame_num in range(0,200):
			ret, img = cap.read()
			if ret:
				for idx, row in test_data_df[(test_data_df['preds'] == 1.0) & (test_data_df['frame_num'] == frame_num)].iterrows():
					cv2.circle(img,(int(row['x']),int(row['y'])),5,(0,0,255),-1)
				video.write(cv2.resize(img, (960, 540)))
		video.release()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Waggle Detector')
	parser.add_argument('--video','-v', help='video for detection')
	parser.add_argument('--model', '-m', default='model.pickle', help='model to use for prediction')
	parser.add_argument('--destination', '-d', help='destination of resulting video')
	args = parser.parse_args()

	detector = Main(args.video, args.model, args.destination)
	detector.detect()
