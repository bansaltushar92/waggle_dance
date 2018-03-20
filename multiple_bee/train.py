from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import numpy as np
import argparse
import glob
from datagen import *
import pandas as pd

class Main(object):

	def __init__(self, video_dir, annotation_dir, model_file):
		self.video_dir = video_dir
		self.annotation_dir = annotation_dir
		self.model_file = model_file

	def train(self):
		
		train_df = []
		video_id = 0
		
		for video in glob.glob(self.video_dir + '*'):
			video_name = video.replace('/','.').split('.')[-2]
			vid_df, vid_labels = Datagen(video_id, video, self.annotation_dir + video_name + '_annot.csv').generate_data()
			train_df.append(vid_df)
			if video_id == 0:
				labels = vid_labels
			else:
				labels  = np.append(labels, vid_labels)
			video_id+=1

		clf_xgb = XGBClassifier(max_depth=3, learning_rate=0.2)
		clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=0, class_weight='balanced')
		clf_xgb.fit(np.array(train_df[0]), labels)
		clf_rf.fit(np.array(train_df[0]), labels)
		model = (clf_xgb, clf_rf)

		filehandler = open(self.model_file,"wb")
		pickle.dump(model,filehandler)
		filehandler.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Waggle Detector')
	parser.add_argument('--video','-v', help='video directory')
	parser.add_argument('--model', '-m', help='file for saving the model')
	parser.add_argument('--annotation', '-a', help='annotation directory')
	args = parser.parse_args()

	trainer = Main(args.video, args.annotation, args.model)
	trainer.train()