import cv2
import numpy as np
import nms
import os
import argparse


def generateBox(imagePath, destDir):
    img = cv2.imread(imagePath, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # mser，最大稳定极值区域
    mser = cv2.MSER_create()

    # detect regions in gray scale image
    regions, _ = mser.detectRegions(img_gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # vis = img.copy()
    keep = []

    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)

        # 过滤到较大字体的boxing，赖茅，仁酒识别受影响
        if (w > 10 and w < 100 and h > 10 and h < 200):
            # pass
            # cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)
            keep.append([x, y, x + w, y + h])

    # cv2.imshow('hulls', vis)
    # cv2.waitKey(0)

    orig = img.copy()
    keep2 = np.array(keep)
    pick = nms.non_max_suppression_fast(keep2, 0.5)
    # imgNameType = imageUrl.split('//')[-1]
    imgNameType = os.path.basename(imagePath)
    imgName = imgNameType.split('.')[0]

    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 0, 0), 1)
        cropped = img_gray[startY:endY, startX:endX]
        cv2.imwrite('%s//%s-%d-%d-%d-%d.jpg' % (destDir, imgName, startX, startY, endX, endY), cropped)

    # cv2.imshow('after nms', orig)
    # cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--images_path', type=str, default='e://maotaiSet//trainval-3',
        help='path of images be boxed, default ' + 'e://maotaiSet//trainval-3'
    )

    parser.add_argument(
        '--boxes_path', type=str, default='e://maotaiSet//trainval-4',
        help='path of images be boxed, default ' + 'e://maotaiSet//trainval-4'
    )


    FLAGS = parser.parse_args()

    imagesPath = FLAGS.images_path
    boxesPath = FLAGS.boxes_path
    print(imagesPath)
    print(boxesPath)

    imageList = os.listdir(imagesPath)
    for image in imageList:
        imagePath = os.path.join(imagesPath,image)
        # print(imagePath)
        generateBox(imagePath, boxesPath)




# #贵州茅台
# img = cv2.imread('e://maotaiSet//trainval-3//000201.jpg', 1)
#
# #茅台王子
# #img = cv2.imread('000184.jpg', 1)
# print(img.shape)
#
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img_gray.shape)
# #自适应二值法
# # th3 = cv2.adaptiveThreshold(img_gray, 255, cv2.DAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
# cv2.imshow('binary', img_gray)
# cv2.waitKey(0)
#
# #mser，最大稳定极值区域
# mser = cv2.MSER_create()
#
#
# #detect regions in gray scale image
# regions, _ = mser.detectRegions(img_gray)
# # print(regions)
# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
# # print(hulls)
#
# vis = img.copy()
# keep = []
#
# for c in hulls:
#     x, y, w, h = cv2.boundingRect(c)
#
#     #过滤到较大字体的boxing，赖茅，仁酒识别受影响
#     if(w > 10 and w < 100 and h > 10 and h < 200):
#         # pass
#         cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)
#         keep.append([x, y, x + w, y + h])
#         # cropped = img[x:x+w, y:y+h]
#         # cv2.imshow('cropped',cropped)
#         # cv2.waitKey(0)
#
# print('%d bounding box' % len(keep))
# cv2.polylines(vis, hulls, 1, (255, 0, 0))

# cv2.imshow('hulls', vis)
# cv2.waitKey(0)
#
# orig = img.copy()
# keep2 = np.array(keep)
#
# pick = nms.non_max_suppression_fast(keep2, 0.5)
# print('after applying non-maximum, %d bounding box' % len(pick))
#
# for (startX, startY, endX, endY) in pick:
#     cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 0, 0), 1)
#     cropped = img_gray[startY:endY, startX:endX]
#     cv2.imwrite('e://maotaiSet//trainval-4//%s-%d-%d-%d-%d.png' % ('test', startX, startY, endX, endY), cropped)

# cv2.imshow('after nms', orig)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# x = 3
# cropped = img_gray[pick[x][1]:pick[x][3], pick[x][0]:pick[x][2]]
# print(pick[x])
# cv2.imwrite('e://maotaiSet//trainval-3//%s-%d-%d-%d-%d.png' % ('test', pick[x][0], pick[x][1], pick[x][2], pick[x][3]), cropped)




# mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
#
# for contour in hulls:
#     cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
#
# text_only = cv2.bitwise_and(img, img, mask=mask)
#
# cv2.imshow("text only", text_only)
#
# cv2.waitKey(0)


#自适应二值法
# th3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
# cv2.imshow('009115.jpg', th3)

#canny边界搜索
# edges = cv2.Canny(img_gray, 50, 150, apertureSize=5)

# lines = cv2.HoughLines(th3, 1, np.pi/180, 520)
# # print(lines)
# for i in range(0,len(lines)):
#     rho, theta = lines[i][0][0],lines[i][0][1]
#     # print(rho)
#     # print(theta)
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
# cv2.imshow('000648.jpg', th3)


# if cv2.waitKey() == 27:
#     cv2.destroyAllWindows()


