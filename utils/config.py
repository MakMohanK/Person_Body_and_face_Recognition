
# initialize the list of class labels MobileNet SSD was trained to

CLASSES = [
    "background", 
    "aeroplane", 
    "bicycle", 
    "bird", 
    "boat",
	"bottle", 
    "bus", 
    "car", 
    "cat", 
    "chair", 
    "cow", 
    "diningtable",
	"dog", 
    "horse", 
    "motorbike", 
    "person", 
    "pottedplant", 
    "sheep",
	"sofa", 
    "train", 
    "tvmonitor"
    ]

CAMERA_INPUT = 0


PROTO = './models/MobileNetSSD_deploy.prototxt.txt'
CAFFE = './models/MobileNetSSD_deploy.caffemodel'