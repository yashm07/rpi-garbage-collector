import cv2
import numpy as np
import os
import time
from tflite_runtime.interpreter import Interpreter

# setting up path variables
MODEL_PATH = os.getcwd() + "/Sample_TFLite_model/detect.tflite"
LABELS_PATH = os.getcwd() + "/Sample_TFLite_model/labelmap.txt"

# reading from coco labels 
file = open(LABELS_PATH, "r") 
classes = [l.strip() for l in file.readlines()]
file.close()

# building interpreter and allocating tensors
interpreter = Interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# retrieving model input and output details
input_info = interpreter.get_input_details()
output_info = interpreter.get_output_details()

# setting up variables for input info
model_height, model_width = input_info[0]["shape"][1:3]
tensor_index = input_info[0]["index"]

# setting up variables for output info
confidences_index = output_info[2]["index"]
classes_index = output_info[1]["index"]
boxes_index = output_info[0]["index"]

# starting camera
cap = cv2.VideoCapture(0)

start_time = time.time()
frame_num = 0

while True:

    # reading from Pi camera
    _, frame = cap.read()
    frame_num += 1

    # retrieving frame dimensions
    org_height, org_width, channels = frame.shape

    # must convert from BGR to RGB, resize to model dimensions
    frame_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_input = cv2.resize(frame_input, (model_width, model_height))

    # model expects (1, h, w, channels)
    input_data = np.expand_dims(frame_input, axis=0)
    
    # forward through model
    interpreter.set_tensor(tensor_index, input_data)
    interpreter.invoke()

    # retrieving outputs
    confidences = interpreter.get_tensor(confidences_index)[0]
    class_ids = interpreter.get_tensor(classes_index)[0]
    boxes = interpreter.get_tensor(boxes_index)[0]

    # looping through detected objects
    for i in range(len(confidences)):

        # only keep if higher than threshold
        if confidences[i] > 0.5:

            # retrieving coordinates, rescaling to image size
            y_max = int(min(720, boxes[i][2] * org_height))
            x_max = int(min(1280, boxes[i][3] * org_width))

            y_min = int(max(1, boxes[i][0] * org_height))
            x_min = int(max(1, boxes[i][1] * org_width))

            # displaying label 
            label = classes[int(class_ids[i])]
            cv2.putText(frame, label + " " + str(round(confidences[i]*100, 2)) + "%",
                        (x_min, y_min - 15), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 2)

            # drawing rectangle
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (255, 0, 0), 2)
            
    # displaying fps
    elapsed_time = time.time() - start_time
    frame_rate = str(round(frame_num / elapsed_time, 2))

    cv2.putText(frame, "FPS: " + frame_rate, (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    # display all results
    cv2.imshow("Object Detector", frame)

    # to exit, 1 for continuous live stream
    if cv2.waitKey(1) == ord("q"):
        break

# closing camera and window
cap.release()
cv2.destroyAllWindows()
