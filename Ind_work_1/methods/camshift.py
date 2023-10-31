import numpy as np
import cv2 as cv


class OurCamShift:
    def start(self):
        cap = cv.VideoCapture(0)
        # take first frame of the video
        ret, frame = cap.read()

        # setup initial location of window
        x, y, w, h = cv.selectROI(frame)
        track_window = (x, y, w, h)
        # set up the ROI for tracking
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        # cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        roi_track_window = self.bbox_to_roi(track_window)
        while (1):
            ret, frame = cap.read()
            if ret:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                # apply camshift to get the new location
                # roi = self.bbox_to_roi(track_window)

                # track_window = self.roi_to_bbox(roi)
                # ret, _ = cv.CamShift(dst, track_window, term_crit)
                roi_track_window = self.camshift(dst, roi_track_window)
                # Draw it on image
                # pts = cv.boxPoints(ret)
                # pts = np.int0(pts)
                # img2 = cv.polylines(frame, [pts], True, 255, 2)

                rec = roi_track_window
                frame = cv.rectangle(frame, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 5)
                cv.imshow('frame', frame)
                cv.imshow('frame1', self.get_area_from_roi(dst, rec))
                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break
            else:
                break

    # ok
    def get_area_from_roi(self, frame, roi):
        x0 = roi[0]
        y0 = roi[1]
        x1 = roi[2]
        y1 = roi[3]

        area = np.empty([y1 - y0, x1 - x0])
        for i in range(y0, y1):
            for j in range(x0, x1):
                area[i - y0][j - x0] = frame[i][j]
        return area

    def compute_new_roi(self, roi, frame, center):
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

        if (x0 < 0):
            x0 = 0
            x1 = int(half_x_len * 2)
        if (y0 < 0):
            y0 = 0
            y1 = int(half_y_len * 2)
        if (y1 > clos):
            y1 = clos
            y0 = int(half_y_len * 2)
        if (x1 > rows):
            x1 = rows
            x0 = int(half_x_len * 2)

        return (x0, y0, x1, y1)

    # (x,y,w,h) to (x0,y0,x1,y1) ok
    def bbox_to_roi(self, bbox):
        return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

    # (x0,y0,x1,y1) to (x,y,w,h) ok
    def roi_to_bbox(self, roi):
        return (roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1])

    def camshift(self, frame, roi):
        iter_count = 0
        centroids_history = list()

        while True:
            a = self.get_area_from_roi(frame, roi)

            moments = cv.moments(a, 0)
            m01 = moments['m01']
            m10 = moments['m10']
            m00 = moments['m00']

            if (m00 == 0):
                break
            xc = m10 / m00
            yc = m01 / m00
            new_roi = self.compute_new_roi(roi, frame, (yc, xc))
            centroids_history.append((yc + roi[0], xc + roi[1]))

            if ((abs(new_roi[0] - roi[0]) < 2 and abs(new_roi[1] - roi[1]) < 2) or iter_count > 5):
                # s = 2 * math.sqrt(m00/255)
                # i1 = new_roi[0]
                # j1 = new_roi[1]
                # i2 = int(s * 1.5)
                # j2 = int(s)

                break

            roi = new_roi
            iter_count += 1
        return roi


camShift = OurCamShift()
camShift.start()
