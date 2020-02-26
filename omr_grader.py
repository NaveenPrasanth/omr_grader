# USAGE
# python test_grader.py --image images/test_01.png

# import the necessary packages

from imutils import contours

import numpy as np
import argparse
import imutils
import cv2
import argparse



def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

# read image from system
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
        help='path to the input image')
args = vars(ap.parse_args())
args = vars(ap.parse_args())
# img = cv2.imread("/home/naveen/optical-mark-recognition/test4.jpg")
img = cv2.imread(args['image'])

print('Original Dimensions : ', img.shape)
scale_percent = 100  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

image = resized

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(len(cnts))
'''
Use the below code to display image
cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Original", image)
cv2.waitKey(5000)
cv2.destroyAllWindows()

'''
documentCnt = None
# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #print(cnts.type)
    # loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break

print(len(cnts))
'''
cv2.drawContours(image, cnts[4:5], -1, (0, 255, 0), 3)
cv2.imshow("Original", image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

'''

# choose the fifth biggest contour from the list since that has all the answers

docCnt = cnts[4]
print(docCnt.shape)
x, y, w, h = cv2.boundingRect(docCnt)
answers_coded = image[y:y+h, x:x+w]

# cuts out the border to make use RETR_EXTERNAL and find all the relevant contours

answers_coded = answers_coded[10:-10, 10:-10]
print(type(answers_coded))

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions

gray_answers = cv2.cvtColor(answers_coded, cv2.COLOR_BGR2GRAY)
blurred_answers = cv2.GaussianBlur(gray_answers, (5, 5), 0)
edged_answers = cv2.Canny(blurred_answers, 75, 200)

thresh = cv2.threshold(blurred_answers, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# cv2.imshow("Original", thresh)
# cv2.waitKey(2000)
# cv2.destroyAllWindows()


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sorting the contours left to right to get cnts in proper order
cnts = contours.sort_contours(cnts, method="left-to-right")[0]


questionCnts = []

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

# in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

correct = 0

# display the found bubbles with numbering over it just for reference
for i,q in enumerate(questionCnts):
    x, y, w, h = cv2.boundingRect(q)
    cv2.putText(answers_coded, str(i), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# use a dummy key for demo

state_alternator = "left"
left_counter = 0
right_counter = 25

dummy_answer_key = {}
for i in range(25):
    dummy_answer_key[i] = 0

for i in range(25,50):
    dummy_answer_key[i] = 3


# looping over questions in a batch of 5 since there are 5 options

for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # sort the contours for the current question from
    # left to right, then initialize the index of the
    # bubbled answer
    cnts = questionCnts[i:i + 5]
    for i, q in enumerate(questionCnts):
        x, y, w, h = cv2.boundingRect(q)
        cv2.putText(answers_coded, str(i), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    bubbled = None

    # loop over the sorted contours
    for (j, c) in enumerate(cnts):
        # construct a mask that reveals only the current
        # "bubble" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # apply the mask to the thresholded image, then
        # count the number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        # if the current total has a larger number of total
        # non-zero pixels, then we are examining the currently
        # bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # initialize the contour color and the index of the
    # *correct* answer
    color = (0, 0, 255)
    if state_alternator == "left":
        k = dummy_answer_key[left_counter]
        left_counter += 1
        state_alternator = "right"
    else:
        k = dummy_answer_key[right_counter]
        right_counter += 1
        state_alternator = "left"

    # check to see if the bubbled answer is correct
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    # draw the outline of the correct answer on the test
    cv2.drawContours(answers_coded, [cnts[k]], -1, color, 3)

# grab the test taker
score = (correct / len(dummy_answer_key)) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(answers_coded, "{:.2f}%".format(score), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", answers_coded)
cv2.waitKey(0)
