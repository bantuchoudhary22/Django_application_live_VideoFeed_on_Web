from pyimagesearch.centroidtracker_mine import CentroidTracker
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
face_detection_videocam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))

#(H, W) = (None, None)
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('opencv_haarcascade_data/deploy.prototxt', 'opencv_haarcascade_data/res10_300x300_ssd_iter_140000.caffemodel')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('opencv_haarcascade_data/trainer/trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX


id = 0
names=['None']
#data = dict()
with open("opencv_haarcascade_data/main.txt",'r') as raw_data:
    for item in raw_data:
        if ':' in item:
            key,value = item.split(':', 1)
            names.append(value)
        else:
            pass


class VideoCamera(object):

	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):

		success, frame = self.video.read()
		#print(frame)
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		H = None
		W = None
		#print(W,H)
		if W is None or H is None:
			H, W = frame.shape[:2]

		blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		rects = []

		for i in range(0, detections.shape[2]):
			if detections[0, 0, i, 2] > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				rects.append(box.astype("int"))

				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
				frame1=frame[startY-40:endY+40,startX-40:endX+40]#ROI
				frame2=frame1.copy()
				frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
				id, confidence = recognizer.predict(frame1[startY:startY+endY,startX:startX+endX])
				if (confidence < 100):
					id = names[id]
					confidence = "  {0}%".format(round(100 - confidence))
				else:
					id = "unknown"
					confidence = "  {0}%".format(round(100 - confidence))
					#cou=cou+1
			
				cv2.putText(frame, str(id), (startX+5,startY-5), font, 1, (255,255,255), 2)
				cv2.putText(frame, str(confidence), (startX+5,startY+endY-5), font, 1, (255,255,0), 1)
				#frame_flip = cv2.flip(frame,1)
				ret, jpeg = cv2.imencode('.jpg', frame)
				return jpeg.tobytes()

		#print("Done now")
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		#for (x, y, w, h) in faces_detected:
		#	cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
	