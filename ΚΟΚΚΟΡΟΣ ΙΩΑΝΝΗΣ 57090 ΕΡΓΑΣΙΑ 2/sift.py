import cv2
import numpy as np
import time


start = time.time() #Έναρξη χρονομέτρησης
print("Start")

"""
ena = cv2.imread('termaaristera.jpg', cv2.COLOR_BGR2GRAY)
dyo = cv2.imread('aristera.jpg', cv2.COLOR_BGR2GRAY)
tria = cv2.imread('deksia.jpg', cv2.COLOR_BGR2GRAY)
tessera = cv2.imread('termadeksia.jpg', cv2.COLOR_BGR2GRAY)
leftest=cv2.cvtColor(ena, cv2.COLOR_BGR2GRAY)
left =cv2.cvtColor(dyo, cv2.COLOR_BGR2GRAY)
right =cv2.cvtColor(tria, cv2.COLOR_BGR2GRAY)
rightest =cv2.cvtColor(tessera, cv2.COLOR_BGR2GRAY)
"""
leftest = cv2.imread('hotel-03.png', cv2.COLOR_BGR2GRAY)
left = cv2.imread('hotel-02.png', cv2.COLOR_BGR2GRAY)
right = cv2.imread('hotel-01.png', cv2.COLOR_BGR2GRAY)
rightest = cv2.imread('hotel-00.png', cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(rightest, None) #δημιουργία keypoints και descriptors με βάση τα keypoints
kp2, des2 = sift.detectAndCompute(right, None)
kp3, des3 = sift.detectAndCompute(left, None)
kp4, des4 = sift.detectAndCompute(leftest, None)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
match = cv2.FlannBasedMatcher(index_params, search_params) #διαφορετική υλοποιήση BFmatcher()
matches = match.knnMatch(des1, des2, k=2) #αντιστοίχιση σημείων
matches2 = match.knnMatch(des3, des4, k=2)

good1 = []
good2 = []

for m, n in matches:
    if m.distance < 0.75*n.distance: #βελτιστοποίηση συνάρτησης για αποφυγή αστοχιών
        good1.append(m)
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
left2 = cv2.drawMatches(rightest, kp1, right, kp2, good1, None, **draw_params) #Σχεδιασμός αντιστοίχισης των σημειων

for m, n in matches2:
    if m.distance < 0.75 *n.distance:
        good2.append(m)
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
right2 = cv2.drawMatches(left, kp3, leftest, kp4, good2, None, **draw_params)


MIN_MATCH_COUNT = 10


if len(good1) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good1]).reshape(-1, 1, 2)  #δημιουργία πρώτου κομματιού εικόνας με βάση τα matches
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good1]).reshape(-1, 1, 2)  #δημιουργία δευετρου κομματιού εικόνας με βάση τα matches
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # Βρίσκει πώς πρέπει να μετατραπει η πρώτη για να "ταιριάξει" με τη δευτερη
    h, w = rightest.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
else:
    print("Not enough matches are found", (len(good1)))

dst = cv2.warpPerspective(rightest, M, (rightest.shape[1] + rightest.shape[1], right.shape[0]))  #warping για σωστότερη αντιστοίχιση εικόνων και αύξηση pixels για να χωράει το output
dst[0:rightest.shape[0], 0:rightest.shape[1]] = right #τοποθέτηση των στοιχείων της εικόνας που βρίσκεται πιο δεξια στη θέση που τους αντιστοιχεί
"""
#Ακολοθει μεθοδολογια για περικοπη του μαυρου τμηματος της εικονας μετα την συχγωνευση με error στην τελικη ενωση could not broadcast input array from shape (768,1639) into shape (768,1546)
_, thresh = cv2.threshold(dst, 1, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #εύρεση contours,δηλαδή καμπυλών που αποτελούν τα όρια μεταξύ εικόνας και μαύρων επιφανειών
cnt = contours[0]
x, y, w, h = cv2.boundingRect(cnt) #ορθογώνιο περίγραμμα γύρω από τα contours
dst = dst[y:y + h, x:x + w] #περικοπή μαύρων επιφανειών
"""
cv2.imwrite('output1.jpg', dst)



if len(good2) > MIN_MATCH_COUNT:
    src_pts2 = np.float32([kp3[m.queryIdx].pt for m in good2]).reshape(-1, 1, 2)
    dst_pts2 = np.float32([kp4[m.trainIdx].pt for m in good2]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)
    h, w = leftest.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)


else:
    print("Not enough matches are found", (len(good2)))

dst2 = cv2.warpPerspective(left, M, (left.shape[1] + left.shape[1], leftest.shape[0]))
dst2[0:left.shape[0], 0:left.shape[1]] = leftest
"""
_, thresh = cv2.threshold(dst, 1, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #εύρεση contours,δηλαδή καμπυλών που αποτελούν τα όρια μεταξύ εικόνας και μαύρων επιφανειών
cnt = contours[0]
x, y, w, h = cv2.boundingRect(cnt) #ορθογώνιο περίγραμμα γύρω από τα contours
dst = dst[y:y + h, x:x + w] #περικοπή μαύρων επιφανειών
"""
cv2.imwrite('output2.jpg', dst2)


finalleft = cv2.imread('output2.jpg', cv2.COLOR_BGR2GRAY)
finalright = cv2.imread('output1.jpg', cv2.COLOR_BGR2GRAY)


kp5, des5 = sift.detectAndCompute(finalleft, None)
kp6, des6 = sift.detectAndCompute(finalright, None)

matches3 = match.knnMatch(des5, des6, k=2)

good3 = []

for m, n in matches3:
    if m.distance < 0.7*n.distance:
        good3.append(m)
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
final = cv2.drawMatches(finalleft, kp5, finalright, kp6, good3, None, **draw_params)


if len(good3) > MIN_MATCH_COUNT:
    src_pts3 = np.float32([kp3[m.queryIdx].pt for m in good2]).reshape(-1, 1, 2)
    dst_pts3 = np.float32([kp4[m.trainIdx].pt for m in good2]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts3, dst_pts3, cv2.RANSAC, 5.0)
    h, w = finalleft.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)


else:
    print("Not enough matches are found", (len(good2)))

dst3 = cv2.warpPerspective(finalright, M, (finalright.shape[1] + finalright.shape[1], finalleft.shape[0]))
dst3[0:finalright.shape[0], 0:finalright.shape[1]] = finalleft
cv2.imwrite('output3.jpg', dst3)
#cv2.imshow('main', dst3)
#cv2.waitKey(0)
end = time.time() #τέλος χρονομέτρησης
print('End')
print(end - start)



