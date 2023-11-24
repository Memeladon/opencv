import math
import time

import numpy as np
import cv2


class CamShiftHandMade:
    def start(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        bbox = cv2.selectROI(frame)
        x, y, w, h = bbox
        track_window = (x, y, w, h)

        roi = frame[y:y + h, x:x + w]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # get histogram for [0] blue, [1] green, [2] red channel
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        # convert hist values 0-180 to a range between 0-1
        roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        roi_track_window = self.bbox_to_roi(track_window)

        prev_frame_time = 0

        while True:
            ret, frame = cap.read()
            if ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                track_window, center, axes, angle = self.mean_shift(dst, roi_track_window)

                # x, y, w, h = track_window
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 4)
                cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 0, 255), 2)

                # Для примера, вывод FPS на изображение
                font = cv2.FONT_HERSHEY_SIMPLEX
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = str(int(fps))

                cv2.putText(frame, fps, (3, 30), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

                cv2.imshow("Camshift Track", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                break

    # ok
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
    def bbox_to_roi(bbox):
        return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

    @staticmethod
    def roi_to_bbox(roi):
        return roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]

    @staticmethod
    def get_area_from_roi(frame, roi):
        x0 = roi[0]
        y0 = roi[1]
        x1 = roi[2]
        y1 = roi[3]

        area = np.empty([y1 - y0, x1 - x0])
        for i in range(y0, y1):
            for j in range(x0, x1):
                area[i - y0][j - x0] = frame[i][j]
        return area

    def mean_shift(self, frame, roi):
        iter_count = 0
        centroids_history = list()

    def mean_shift(self, frame, roi):
        iter_count = 0
        centroids_history = list()

        while True:
            area = self.get_area_from_roi(frame, roi)

            moments = cv2.moments(area, 0)
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

        # Вычисление параметров эллипса: центр, оси, угол наклона
        center = (roi[0] + roi[2] // 2, roi[1] + roi[3] // 2)
        axes = (roi[2] // 2, roi[3] // 2)
        angle = 0  # Угол наклона, т.к. камера смотрит прямо, угол = 0

        return roi, center, axes, angle


video_path = '../videos/source/1.mp4'
camShift = CamShiftHandMade()
camShift.start(video_path)
