# Importing the required libraries.
import numpy as np
import cv2
import time

# Setting up working directory .
work_dir = 'images\\'

# Reading the image using OpenCV .
image_BGR = cv2.imread(work_dir+'women-looking-financial-documents.jpg')
# Showing up the original image in named window.
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", image_BGR)
cv2.waitKey(0)
cv2.destroyWindow("Original Image")
# Taking out height and width of original image.
h, w = image_BGR.shape[:2]
# Converting original image to BLOB.
blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416,416), swapRB=True, crop=False)
# Slicing the BLOB and transposing it to BGR to show the blob image.
blob_to_show = blob[0, :, :, :].transpose(1,2,0)
blob_to_show = cv2.cvtColor(blob_to_show, cv2.COLOR_RGB2BGR)
cv2.namedWindow("BLOB Image", cv2.WINDOW_NORMAL)
cv2.imshow("BLOB Image", blob_to_show)
cv2.waitKey(0)
cv2.destroyWindow("BLOB Image")
# lOADING LABELS FILE. YOLO COCO data set has predefined classes of objects. Just extracting
# them
with open ('yolo-coco-data\coco.names') as f:
    labels = [line.strip() for line in f]

# Creating random colors for each class. It will be used to creating boundaries
# in different colours for each unique object.
colours = np.random.randint(0, 255, size=(len(labels),3), dtype='uint8')

# Creating YOLO network.
network = cv2.dnn.readNetFromDarknet('yolo-coco-data\yolov3.cfg','yolo-coco-data\yolov3.weights')
layers_name_all = network.getLayerNames()

# Taking out only unconnected layers names. As we are using the pretrained model of YOLO
# we are just interested in output layers so extracting the name of those layers.
layers_names_output = [layers_name_all[i[0] -1] for i in network.getUnconnectedOutLayers()]
# Setting up probability & threshold to filter out detected objects.
probability_minimum = 0.5
threshold = 0.3
# setting input to network
network.setInput(blob)
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()
print('Object Detection took {:.4f} seconds'.format(end - start))
# Preparing lists for detected bounding boxes,obtained confidences and class's number
bounding_boxes = []
confidences = []
class_numbers = []
# Going through the results.
for result in output_from_network:
    for detected_objects in result:
        scores = detected_objects[5:]
        current_class = np.argmax(scores)
        current_confidence = scores[current_class]
        if current_confidence > probability_minimum:
            box_current = detected_objects[0:4] * np.array([w,h,w,h])
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width/2))
            y_min = int(y_center - (box_height/2))
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(current_confidence))
            class_numbers.append(current_class)
# Implementing non-maximum suppression of given bounding boxes
# With this technique we exclude some of bounding boxes if their
# corresponding confidences are low or there is another
# bounding box for this region with higher confidence

# It is needed to make sure that data type of the boxes is 'int'
# and data type of the confidences is 'float'
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,probability_minimum, threshold)
# Defining counter for detected objects
counter = 1

# Checking if there is at least one detected object after non-maximum suppression
if len(results) > 0:
    # Going through indexes of results
    for i in results.flatten():
        # Showing labels of the detected objects
        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))
        # Incrementing counter
        counter += 1
        # Getting current bounding box coordinates,
        # its width and height
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
        # Preparing colour for current bounding box
        # and converting from numpy array to list
        colour_box_current = colours[class_numbers[i]].tolist()
        # Drawing bounding box on the original image
        cv2.rectangle(image_BGR, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)
        # Preparing text with label and confidence for current bounding box
        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])
        # Putting text with label and confidence on the original image
        cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

# Comparing how many objects where before non-maximum suppression
# and left after
print()
print('Total objects been detected:', len(bounding_boxes))
print('Number of objects left after non-maximum suppression:', counter - 1)

# Showing Original Image with Detected Objects
# Giving name to the window with Original Image
# And specifying that window is resizable
cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
# Pay attention! 'cv2.imshow' takes images in BGR format
cv2.imshow('Detections', image_BGR)
# Waiting for any key being pressed
cv2.waitKey(0)
# Destroying opened window with name 'Detections'
cv2.destroyWindow('Detections')
