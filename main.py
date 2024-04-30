# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os
import pygame
# Local configuration
from utils.config import CLASSES, CAMERA_INPUT, PROTO, CAFFE, SOUND, LABEL, MODEL

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model(MODEL, compile=False)
class_names = open(LABEL, "r").readlines()

# detect, then generate a set of bounding box colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO]..... loading model...")
net = cv2.dnn.readNetFromCaffe(PROTO, CAFFE)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO]..... starting video stream...")
vs = VideoStream(src=CAMERA_INPUT).start()
time.sleep(2.0)
fps = FPS().start()


def play_sound():
	pygame.init()
	pygame.mixer.music.load(SOUND)
	pygame.mixer.music.play()
	while pygame.mixer.music.get_busy():
		pygame.time.Clock().tick(10)
	pygame.quit()


# play_sound()

def count_files_in_folder(folder_path):
    num_files = 0
    for _, _, files in os.walk(folder_path):
        num_files += len(files)
    return num_files

def create_database(img, path):
	count = count_files_in_folder(path)
	img = cv2.resize(img, (300, 300))
	name = path+"img_"+str(count)+".jpg"
	cv2.imwrite(name, img)


def save_to_database(img, clas): # this will save the images in folder
	if clas == 'person':
		path = "./database/person/kalyan/"
		recognise(img)
		# create_database(img, path)

	elif clas == 'aeroplane':
		pass

	elif clas == 'boat':
		pass

	elif clas == 'bus':
		pass

	elif clas == 'car':
		pass

	elif clas == 'motorbike':
		pass



def recognise(image):
	try:
		data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
		image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
		image_array = np.asarray(image)
		normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
		data[0] = normalized_image_array
		prediction = model.predict(data)
		index = np.argmax(prediction)
		class_name = class_names[index]
		confidence_score = prediction[0][index]
		print("Class:", class_name[2:], end="")
		print("Confidence Score:", confidence_score)
	except:
		print("[INFO].... No object detected")



def main():
	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > 0.4:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				crop_img = frame[startY:endY, startX:endX]
				try:
					save_to_database(crop_img, CLASSES[idx])
					cv2.imshow("CROP", crop_img)
					# draw the prediction on the frame
					label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
					cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				except:
					print("[INFO].... Unable to crop the image")

		# show the output frame
		cv2.imshow("OUTPUT", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()


if __name__ == "__main__":
	main()

# stop the timer and display FPS information
fps.stop()
print("[INFO]..... elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO]..... approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()