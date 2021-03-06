import sys
import cv2
import math
import numpy as np
import pandas as pd
from settings_local import *

def calc_dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    """
    Computes the nearest point on a line segment from the given point
    
    :param int x1: x coordinate of point-1 of line segment
    :param int x1: x coordinate of point-1 of line segment
    :param int x1: x coordinate of point-1 of line segment
    :param int x1: x coordinate of point-1 of line segment
    :param int x1: x coordinate of point-1 of line segment
    :param int x1: x coordinate of point-1 of line segment
    
    :returns: distance and coordinates of the closest point on line segment
    
    """
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = math.sqrt(dx*dx + dy*dy)

    return dist, int(x), int(y)

def select_point(event, x, y, flags, param):
    """
    Callback function that assigns fetches the cursor coordinates on the image
    
    """
    global prev
    global abdomen
    
    if prev == None and event == cv2.EVENT_LBUTTONUP:
        prev = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        abdomen = (x,y)

def find_orient(img_gray, prev_orient):
    """
    Computes the orientation of the bee in given frame
    
    :param img_gray: grayscale crooped image of the bee
    :param prev_orient: orientation in the previous frame
    :returns: orientation of the bee (degrees) and rectangular mask to filter bee from original image
    
    """
    max_val = 0
    orient = 0
    ret,img_thresh = cv2.threshold(img_gray,THRESHOLD,255,cv2.THRESH_BINARY)
    temp = np.zeros_like(img_thresh)
    rect = np.zeros_like(img_thresh)
    rows,cols = temp.shape
    cv2.rectangle(rect, (cols/2,rows/2-ELLIPSE_AXIS[1]), ((cols/2-2*ELLIPSE_AXIS[0],rows/2+ELLIPSE_AXIS[1])), 255)
    cv2.ellipse(temp, (cols/2+ELLIPSE_AXIS[0],rows/2), ELLIPSE_AXIS,0, 0, 360, 255,-1)
    rows,cols = temp.shape
    for i in range(prev_orient-15,prev_orient+15,5):
        M = cv2.getRotationMatrix2D((cols/2,rows/2), i,1)
        dst = cv2.warpAffine(img_gray,M,(cols,rows))
        res = cv2.bitwise_and(dst,dst,mask=temp)
        if max_val < np.sum(res):
            max_val = np.sum(res)
            orient = i
    M = cv2.getRotationMatrix2D((cols/2,rows/2), 180-orient,1)
    result = cv2.warpAffine(rect,M,(cols,rows))
    return orient, result

def find_contours(prev,contours):
    """
    Returns the center of nearest contour from given point
    :param int tuple prev: coordinates of previous position (of marker)
    :param list contours: list of contours
    
    """
    
    dis = None
    if len(contours) > 1:
        for cnt in contours:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            a = np.array([prev[0], prev[1]])
            b = np.array([int (x), int (y)])
            if dis == None:
                dis = np.linalg.norm(a-b)
                [new_x,new_y] = b
            elif dis > np.linalg.norm(a-b):
                dis = np.linalg.norm(a-b)
                [new_x,new_y] = b
        return (new_x,new_y), dis 
    return None

def check_zig_zag(track_pos):
    """
    Check if bee motion is zig-zag motion
    :param list track_pos: list of bee positions
    :returns: bool
    
    """
    
    if len(track_pos) > 3:
        (a_x,a_y) = track_pos[-4]
        (b_x,b_y) = track_pos[-3]
        (c_x,c_y) = track_pos[-2]
        (d_x,d_y) = track_pos[-1]
        
        if (b_x-a_x)*(c_x-b_x) + (b_y-a_y)*(c_y-b_y) < 0 and (c_x-b_x)*(d_x-c_x) + (c_y-b_y)*(d_y-c_y) < 0:
            return True
    return False

def tracker(filename):
    
    sharp_list = []
    track_pos = []
    orient_list = []
    global prev
    prev = None
    global abdomen
    
    cap = cv2.VideoCapture(filename)
    for _ in range(0,SKIP_FRAMES):
        ret, img = cap.read()
    ret, img = cap.read()
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cv2.namedWindow("frame")
    cv2.imshow('frame', img)
    cv2.setMouseCallback("frame",  select_point)    
    cv2.waitKey(0)
    
    # Initial orient
    base = np.array([-1,0])
    vect = np.array([prev[0]-abdomen[0],prev[1]-abdomen[1]])
    orient = int(math.degrees(np.arccos(np.dot(base, vect)/(np.linalg.norm(base)*np.linalg.norm(vect)))))
    if vect[1] > 0:
        orient = 360-orient    
    
    mask = np.zeros_like(img)
    waggle_mask = np.zeros_like(img)

    for counter in range(0,FRAME_COUNT):
        ret, img = cap.read()
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
        img_filt = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)    
        img_filt, contours, hierarchy = cv2.findContours(img_filt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        (new_x,new_y), dis = find_contours(prev,contours)    
    
        if dis < 10:
            img = cv2.circle(img,(new_x,new_y),2,(0,0,255),-1)
            mask[:,:,2] = mask[:,:,2].clip(min=2)
            mask[:,:,2] = mask[:,:,2] - 2
            mask = cv2.line(mask, (prev[0],prev[1]),(new_x,new_y), (0,0,255), 2)    
            
            
            prev = (new_x, new_y)
            
            track_pos.append(prev)
            '''Calculating distance from fixed line segment'''
            dist, map_x, map_y = calc_dist(488,317,490,333,new_x,new_y)
            
            
        img = cv2.add(img, mask)

        orient, rect_mask = find_orient(img_gray[new_y-40:new_y+40,new_x-40:new_x+40], orient)
        orient_list.append(orient)

        M = cv2.getRotationMatrix2D((new_x,new_y), orient,1)
        rows,cols = img_gray.shape
        rotated_img = cv2.warpAffine(img_gray,M, (cols, rows))
        bee_rect = rotated_img[new_y-ELLIPSE_AXIS[1]:new_y+ELLIPSE_AXIS[1],new_x:new_x+2*ELLIPSE_AXIS[0]]

        laplacian = cv2.Laplacian(bee_rect,cv2.CV_64F)
        mean, sigma = cv2.meanStdDev(laplacian)
        sharp_list.append(np.log(sigma[0])[0])
    
    cv2.destroyAllWindows()
    cap.release()
    return sharp_list, track_pos, orient_list


def get_peak_list(orient_list):
    min_list = []
    max_list = []
    for i in range(10,len(orient_list)-10):
        if max(orient_list[i-10:i+11]) == orient_list[i]:
            max_list.append(i)
        elif min(orient_list[i-10:i+11]) == orient_list[i]:
            min_list.append(i)
    return min_list, max_list

def compute_max_waggle(max_list, orient_list, waggle_list, convolve_list, threshold):
    for i in max_list:
        j = i
        while j < len(orient_list) and orient_list[i] - orient_list[j] >= 0 and (i == j or np.mean(orient_list[i-1:i+2]) - np.mean(orient_list[i+1:j+1]) <= 10):
            if convolve_list[i] <= threshold:
                waggle_list[j] = True
            j+=1
        j=i-1
        while j >= 0 and orient_list[i] - orient_list[j] >= 0 and np.mean(orient_list[i-1:i+2]) - np.mean(orient_list[j:i]) <= 10:
            if convolve_list[i] <= threshold:
                waggle_list[j] = True
            j-=1
    return waggle_list

def compute_min_waggle(min_list, orient_list, waggle_list, convolve_list, threshold):
    for i in min_list:
        j = i
        while j < len(orient_list) and orient_list[j] - orient_list[i] >= 0 and (i == j or np.mean(orient_list[i+1:j+1]) - np.mean(orient_list[i-1:i+2]) <= 10):
            if convolve_list[i] <= threshold:
                waggle_list[j] = True
            j+=1
        j=i-1
        while j >= 0 and orient_list[j] - orient_list[i] >= 0 and np.mean(orient_list[j:i]) - np.mean(orient_list[i-1:i+2]) <= 10:
            if convolve_list[i] <= threshold:
                waggle_list[j] = True
            j-=1
    return waggle_list

def show_results(filename, waggle_list):
    cap = cv2.VideoCapture(filename)
    for _ in range(0,SKIP_FRAMES):
        ret, img = cap.read()

    height , width =  int(cap.get(4)), int(cap.get(3))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_'+ filename + '.avi',fourcc, 30.0, (width,height))

    for counter in range(0,FRAME_COUNT):
        ret, img = cap.read()
        cv2.putText(img, str(counter),(100,100), FONT, 2,(255,255,255),2,cv2.LINE_AA)
        if waggle_list[counter] == True:
            cv2.putText(img, 'WAGGLE',(width-400,100), FONT, 2,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow('img', img)
        out.write(img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    cap.release()
    out.release()

def compute_waggle(orient_list, sharp_list, filename):
    waggle_list = [False]*len(orient_list)
    convolve_sharp_list = np.convolve(sharp_list, CONVOLVE_ARRAY, 'valid')
    threshold = SHARPNESS_THRESHOLD*(max(convolve_sharp_list)-min(convolve_sharp_list))+min(convolve_sharp_list)
    min_list, max_list = get_peak_list(orient_list)
    waggle_list = compute_min_waggle(min_list, orient_list, waggle_list, convolve_sharp_list, threshold)
    waggle_list = compute_max_waggle(max_list, orient_list, waggle_list, convolve_sharp_list, threshold)
    show_results(filename, waggle_list)


if __name__ == "__main__":
    sharp_list, track_pos, orient_list = tracker(sys.argv[1])
    compute_waggle(orient_list, sharp_list, sys.argv[1])
    