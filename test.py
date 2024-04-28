import cv2

# Replace 'path/to/your/MobileNetSSD_deploy.prototxt' with the actual file path.
prototxt_path = './real-time-object-detection/MobileNetSSD_deploy.prototxt.txt'

# Replace 'path/to/your/MobileNetSSD_deploy.caffemodel' with the actual file path.
caffemodel_path = './real-time-object-detection/MobileNetSSD_deploy.caffemodel'

net = cv2.dnn.readNet(prototxt_path, caffemodel_path)