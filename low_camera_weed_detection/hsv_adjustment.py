import sys
sys.path.append("/Users/machidakentarou/GitHub/stock_2022")

import cv2
import numpy as np

B_THRESH = (0, 255)
G_THRESH = (0, 255)
R_THRESH = (0, 255)
MASKING_TYPE = 'hsv'
AREA_LOW_THRESH_RATE = 0.05
AREA_HIGH_THRESH_RATE = 0.8
ASPECT_LOW_THRESH_RATE = 0.05
ASPECT_HIGH_THRESH_RATE = 2.0
CLOSING = (2, 2)
OPENING = (4, 4)
H_THRESH = (30, 60)
S_THRESH = (5, 90)
V_THRESH = (140, 180)

IMG_PATH = '../data/test_img.png'

# for macOS
H_KEY = 104
S_KEY = 115
V_KEY = 118
R_ARROW = 3
L_ARROW = 2
UP_ARROW = 0
DOWN_ARROW = 1
ESC_KEY = 27

def change_thresh(thresh:int,right_or_left:int)->int:
    """
    左右keyで閾値の操作を行う
    """
    if right_or_left == R_ARROW:
        thresh += 1
    elif right_or_left == L_ARROW:
        thresh -= 1
    return thresh

def get_binary(frame ,h_thresh:tuple, s_thresh:tuple, v_thresh:tuple):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([h_thresh[0], s_thresh[0], v_thresh[0]])
    upper = np.array([h_thresh[1], s_thresh[1], v_thresh[1]])

    binary = cv2.inRange(hsv, lower, upper)
    return binary

if __name__=='__main__':
    while True:
        frame = cv2.imread(IMG_PATH)
        binary = get_binary(frame,H_THRESH,S_THRESH,V_THRESH)
        cv2.imshow("test",binary)
        print(f"{H_THRESH=}, {S_THRESH=}, {V_THRESH=}")
        mode_key = cv2.waitKey(100000000)  # hsvのどれかを選択
        if mode_key == H_KEY:
            print('h mode start.')
            while True:
                high_or_low = cv2.waitKey(100000000)
                if high_or_low == UP_ARROW:
                    print("you can change high thresh.")
                    while True:
                        right_or_left = cv2.waitKey(100000000)
                        if right_or_left != R_ARROW and right_or_left != L_ARROW:
                            print(right_or_left)
                            break
                        H_THRESH = (H_THRESH[0],change_thresh(H_THRESH[1], right_or_left))
                        print(H_THRESH)
                        binary = get_binary(frame,H_THRESH,S_THRESH,V_THRESH)
                        cv2.imshow("test",binary)
    
                elif high_or_low == DOWN_ARROW:
                    print("you can change low thresh.")
                    while True:
                        right_or_left = cv2.waitKey(100000000)
                        if right_or_left != R_ARROW and right_or_left != L_ARROW:
                            print(right_or_left)
                            break
                        H_THRESH = (change_thresh(H_THRESH[0],right_or_left), H_THRESH[1])
                        print(H_THRESH)
                        binary = get_binary(frame,H_THRESH,S_THRESH,V_THRESH)
                        cv2.imshow("test",binary)
                else:
                    print('h mode end.')
                    break
        
        elif mode_key == S_KEY:
            print('s mode start.')
            while True:
                high_or_low = cv2.waitKey(100000000)
                if high_or_low == UP_ARROW:
                    print("you can change high thresh.")
                    while True:
                        right_or_left = cv2.waitKey(100000000)
                        if right_or_left != R_ARROW and right_or_left != L_ARROW:
                            print(right_or_left)
                            break
                        S_THRESH = (S_THRESH[0],change_thresh(S_THRESH[1], right_or_left))
                        print(S_THRESH)
                        binary = get_binary(frame,H_THRESH,S_THRESH,V_THRESH)
                        cv2.imshow("test",binary)

                elif high_or_low == DOWN_ARROW:
                    print("you can change low thresh.")
                    while True:
                        right_or_left = cv2.waitKey(100000000)
                        if right_or_left != R_ARROW and right_or_left != L_ARROW:
                            print(right_or_left)
                            break
                        S_THRESH = (change_thresh(S_THRESH[0], right_or_left), S_THRESH[1])
                        print(S_THRESH)
                        binary = get_binary(frame,H_THRESH,S_THRESH,V_THRESH)
                        cv2.imshow("test",binary)
                else:
                    print('s mode end.')
                    break
                binary = get_binary(frame,H_THRESH,S_THRESH,V_THRESH)
                cv2.imshow("test",binary)
        elif mode_key == V_KEY:
            print('v mode start.')
            while True:
                high_or_low = cv2.waitKey(100000000)
                if high_or_low == UP_ARROW:
                    print("you can change high thresh.")
                    while True:
                        right_or_left = cv2.waitKey(100000000)
                        if right_or_left != R_ARROW and right_or_left != L_ARROW:
                            print(right_or_left)
                            break
                        V_THRESH = (V_THRESH[0],change_thresh(V_THRESH[1], right_or_left))
                        print(V_THRESH)
                        binary = get_binary(frame,H_THRESH,S_THRESH,V_THRESH)
                        cv2.imshow("test",binary)
                elif high_or_low == DOWN_ARROW:
                    print("you can change low thresh.")
                    while True:
                        right_or_left = cv2.waitKey(100000000)
                        if right_or_left != R_ARROW and right_or_left != L_ARROW:
                            print(right_or_left)
                            break
                        V_THRESH = (change_thresh(V_THRESH[0], right_or_left), V_THRESH[1])
                        print(V_THRESH)
                        binary = get_binary(frame,H_THRESH,S_THRESH,V_THRESH)
                        cv2.imshow("test",binary)
                else:
                    print('v mode end.')
                    break
                binary = get_binary(frame,H_THRESH,S_THRESH,V_THRESH)
                cv2.imshow("test",binary)
        else:
            print("Existing mode is h, s, or v")