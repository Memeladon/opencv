from __future__ import print_function
import sys

from Ind_work_1.methods.CamShiftHandMade import CamShiftHandMade

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

    import numpy as np
    import cv2 as cv


class PerfectCamShift(object):
    def __init__(self, video_src):
        self.cam = cv.VideoCapture(0)

        _ret, self.frame = self.cam.read()
        cv.namedWindow('camshift')
        cv.setMouseCallback('camshift', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None
        self.shift = CamShiftHandMade()

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                         (int(180.0 * i / bin_count), 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)

    def run(self):
        while True:
            _ret, self.frame = self.cam.read()
            vis = self.frame.copy()

            hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]

                hist = cv.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
            # if _ret:
                self.selection = None
                dst = cv.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                dst &= mask
                term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

                # track_box, self.track_window = cv.CamShift(dst, self.track_window, term_crit)
                self.track_window, track_box = self.shift.mean_shift(dst, self.track_window)

                x, y, w, h = self.track_window

                if self.show_backproj:
                    vis[:] = dst[..., np.newaxis]
                try:
                    # cv.ellipse(vis, track_box, (0, 0, 255), 2)
                    cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 5)
                except:
                    print(f'track_box: {track_box}')

            cv.imshow('camshift', vis)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv.destroyAllWindows()
