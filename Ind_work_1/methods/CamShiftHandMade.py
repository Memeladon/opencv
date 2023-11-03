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
    def compute_new_roi(roi, new_centroid):
        img_new_centroid = (new_centroid[0] + roi[0], new_centroid[1] + roi[1])
        x = abs(img_new_centroid[0] - (roi[2] / 2))
        y = abs(img_new_centroid[1] - (roi[3] / 2))
        return int(x), int(y), roi[2], roi[3]

    @staticmethod
    def compute_moments(img, i, j):
        moment = 0
        for x, row in enumerate(img):
            for y, pixel in enumerate(row):
                moment = moment + (x ** i * y ** j * pixel)
        return moment

    @staticmethod
    def bbox_to_roi(bbox):
        return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

    @staticmethod
    def roi_to_bbox(roi):
        return roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]

    def camshift(self, prob_distr, roi):
        iter_count = 0
        centroids_history = list()

        while True:
            roi_prob_distr = self.get_area_from_roi(prob_distr, roi)

            m00 = self.compute_moments(roi_prob_distr, 0, 0)
            m10 = self.compute_moments(roi_prob_distr, 1, 0)
            m01 = self.compute_moments(roi_prob_distr, 0, 1)

            if m00 == 0:
                break

            xc = m10 / m00
            yc = m01 / m00
            new_roi = self.compute_new_roi(roi, (yc, xc))

            centroids_history.append((yc + roi[0], xc + roi[1]))

            if ((abs(new_roi[0] - roi[0]) < 2 and
                 abs(new_roi[1] - roi[1]) < 2) or iter_count > 20):
                s = 2 * math.sqrt(m00)
                roi = (new_roi[0], new_roi[1], int(s * 1.5), int(s))
                break

            roi = new_roi
            iter_count += 1
        return roi, centroids_history



