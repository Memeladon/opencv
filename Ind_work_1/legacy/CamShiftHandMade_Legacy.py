import numpy as np
import cv2
import math

MIN_SATURATION = 0.2
MIN_VALUE = 0.5
MIN_PROB = 0.21
nbins = 256


class CamShiftHandMade:

    @staticmethod
    def min_max_scaling(histogram_values):
        mn, mx = histogram_values.min(), histogram_values.max()
        return (histogram_values - mn) / (mx - mn)

    @staticmethod
    def get_area_from_roi(img, roi):
        return img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    @staticmethod
    def compute_new_roi(roi, frame, center):
        x0 = roi[0]
        y0 = roi[1]
        x1 = roi[2]
        y1 = roi[3]

        half_y_len = (y1 - y0) / 2
        half_x_len = (x1 - x0) / 2

        new_y = int(center[0] - half_y_len)
        new_x = int(center[1] - half_x_len)

        x0 = int(x0 + new_x)
        y0 = int(y0 + new_y)
        x1 = int(x1 + new_x)
        y1 = int(y1 + new_y)

        clos, rows = frame.shape

        if x0 < 0:
            x0 = 0
            x1 = int(half_x_len * 2)
        if y0 < 0:
            y0 = 0
            y1 = int(half_y_len * 2)
        if y1 > clos:
            y1 = clos
            y0 = int(half_y_len * 2)
        if x1 > rows:
            x1 = rows
            x0 = int(half_x_len * 2)

        return x0, y0, x1, y1

    @staticmethod
    def compute_moments(img, i, j):
        moment = 0
        max_x, max_y = img.shape[:2]

        for x in range(max_x):
            for y in range(max_y):
                pixel = img[x, y]
                moment = moment + ((x / max_x) ** i * (y / max_y) ** j * pixel)

        return moment

    @staticmethod
    def bbox_to_roi(bbox):
        return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

    @staticmethod
    def roi_to_bbox(roi):
        return roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]

    def myMeanshift(self, frame, roi):
        iter_count = 0
        centroids_history = list()

        while True:
            a = self.get_area_from_roi(frame, roi)

            moments = cv2.moments(a, 0)
            m01 = moments['m01']
            m10 = moments['m10']
            m00 = moments['m00']

            if m00 == 0:
                break
            xc = m10 / m00
            yc = m01 / m00
            new_roi = self.compute_new_roi(roi, frame, (yc, xc))
            centroids_history.append((yc + roi[0], xc + roi[1]))

            if (abs(new_roi[0] - roi[0]) < 2 and abs(new_roi[1] - roi[1]) < 2) or iter_count > 5:
                s = 2 * math.sqrt(m00 / 255)
                roi = (new_roi[0], new_roi[1], int(s * 0.6), int(s * 0.7))

                break

            roi = new_roi
            iter_count += 1
        return roi, roi

