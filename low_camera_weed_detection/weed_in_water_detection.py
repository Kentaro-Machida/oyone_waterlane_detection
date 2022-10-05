import sys
sys.path.append("/Users/machidakentarou/GitHub/stock_2022")

import bbox_searcher
import cv2
import numpy as np

MP4_PATH = "../data/8_30/downward/under_direction_good/output.mp4"
B_THRESH = (0, 255)
G_THRESH = (0, 255)
R_THRESH = (0, 255)
MASKING_TYPE = 'hsv'
AREA_LOW_THRESH_RATE = 0.05
AREA_HIGH_THRESH_RATE = 0.8
ASPECT_LOW_THRESH_RATE = 0.05
ASPECT_HIGH_THRESH_RATE = 2.0
CLOSING = (1, 1)
OPENING = (4, 4)
H_THRESH = (30, 60)
S_THRESH = (5, 90)
V_THRESH = (140, 180)
TEST_IMT_PATH = '../data/test_img.png'


getter = bbox_searcher.Bbox_Getter(
    B_THRESH, G_THRESH, R_THRESH,
    AREA_LOW_THRESH_RATE, AREA_HIGH_THRESH_RATE,
    MASKING_TYPE,
    ASPECT_LOW_THRESH_RATE, ASPECT_HIGH_THRESH_RATE,
    CLOSING, OPENING,
    H_THRESH, S_THRESH, V_THRESH
    )

cap = cv2.VideoCapture(MP4_PATH)
while True:
    ret, frame = cap.read()
    # frame = cv2.imread(TEST_IMT_PATH)
    frame = cv2.resize(frame, (224,224))
    boxes = getter.get_bbox(frame)
    for x1, y1, x2, y2 in boxes:
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_4)
    
    cv2.imshow("test" ,frame)
    # getter.describe_binary(frame)
    key = cv2.waitKey(50)
    if key == 27:
        sys.exit(0)