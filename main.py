"""
Vending machine python, Ziyue Wang
"""

import torch
import cv2
import pandas as pd
import time
from sort import *
import argparse

def main(model_path,input_,max_age,min_hits,iou_threshold):
	model = torch.hub.load('ultralytics/yolov5','custom',path=model_path)
	vid = cv2.VideoCapture(input_)
	mot_tracker = Sort(max_age, min_hits, iou_threshold)

	while True:
		s = time.time()
		ret, frame = vid.read()
		frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		results = model(frame_rgb)
		df = results.pandas().xyxy[0]
		
		if df.empty:
			dets = np.empty((0,5))
			trackers = mot_tracker.update(dets)
			continue

		det_list = []

		for row in df.itertuples():
			if row.confidence < 0.5:
				continue

			trackers = mot_tracker.update(dets)
			det_list.append(np.array([row.xmin,row.ymin,row.xmax,row.ymax,row.confidence]))
		
		if not det_list:
			#If no detections
			dets = np.empty((0,5))
		else:
			#If there are detections
			dets = np.array(det_list)

		trackers = mot_tracker.update(dets)

		for d in trackers:
			print(np.shape(d))
			cv2.rectangle(frame,(int(d[0]),int(d[1])),(int(d[2]),int(d[3])),(0,0,255),2)

		cv2.imshow('',frame)
		print(time.time()-s)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	vid.release()
	cv2.destroyAllWindows()
	print('done')

if __name__ == "__main__:":
	parser = argparse.ArgumentParser(description="Run yolov5 Sort")
	parser.add_argument('--model_path','-m',help='Yolov5 model path')
	parser.add_argument('--input','-i',help='Input source')
	parser.add_argument('--max_age',help='Maximum number of frames to keep alive object without associated box')
	parser.add_argument('--min_hits',help='Minimum number of hits to initiate track')
	parser.add_argument('--iou_thresh',help='Minimum IOU for match')
	args = parser.parse_args()

	main(args.model_path,args.input,args.max_age,args.min_hits,args.iou_thresh)
