#ΟΡΑΣΗ ΥΠΟΛΟΓΙΣΤΩΝ ΕΡΓΑΣΙΑ 1
#ΚΟΚΚΟΡΟΣ ΙΩΑΝΝΗΣ 57090



import cv2
import numpy as np


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


# Load image, grayscale, Otsu's threshold

image = cv2.imread('2_noise.png')

image = cv2.medianBlur(image, 3)

resize = ResizeWithAspectRatio(image, width=720, height=1400, inter=cv2.INTER_AREA)

gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Create rectangular structuring element and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 4))
dilate = cv2.dilate(thresh, kernel, iterations=3)

mask = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) #Νέος κερνελ για την ευρεση λεξεων
dilate2 = cv2.dilate(thresh, mask, iterations=4)

# Find contours and draw rectangle
cnt = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[0] if len(cnt) == 2 else cnt[1]

cnt2 = cv2.findContours(dilate2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt2 = cnt2[0] if len(cnt2) == 2 else cnt2[1]
i = 0

for c in cnt:
    k = 0
    i = i+1
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(resize, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.putText(resize, str(i), (x+1, y+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    gray_intensity = resize.mean(axis=0).mean()
    for l in cnt2:
        k = k + 1
    print('----Region', i, ':----')
    print('Area(px):', )
    print('Bounding Box Area (px):', w * h)
    print('Number of words: ', k)
    print('Mean gray-level value in bounding box:', gray_intensity)

cv2.imshow('image2', resize)
cv2.waitKey(0)






















