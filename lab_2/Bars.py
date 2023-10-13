import numpy as np
import cv2 as cv
import argparse


class Test:
    @classmethod
    def class_work(cls):
        def nothing(x):
            pass

        cv.namedWindow("setup")
        cv.createTrackbar("low_h", "setup", 0, 180, nothing)
        cv.createTrackbar("low_s", "setup", 0, 255, nothing)
        cv.createTrackbar("low_v", "setup", 0, 255, nothing)
        cv.createTrackbar("up_h", "setup", 180, 180, nothing)
        cv.createTrackbar("up_s", "setup", 255, 255, nothing)
        cv.createTrackbar("up_v", "setup", 255, 255, nothing)
        cap = cv.VideoCapture(0)

        while True:
            # Считываем кадр с камеры
            ret, frame = cap.read()

            if not ret:
                break

            low_h = cv.getTrackbarPos('low_h', 'setup')
            low_s = cv.getTrackbarPos('low_s', 'setup')
            low_v = cv.getTrackbarPos('low_v', 'setup')
            up_h = cv.getTrackbarPos('up_h', 'setup')
            up_s = cv.getTrackbarPos('up_s', 'setup')
            up_v = cv.getTrackbarPos('up_v', 'setup')

            min_p = (low_h, low_s, low_v)
            max_p = (up_h, up_s, up_v)
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv_frame, min_p, max_p)

            # Применяем морфологическое преобразование для улучшения результата
            kernel = np.ones((5, 5), np.uint8)
            mask = cv.erode(mask, kernel, iterations=1)
            mask = cv.dilate(mask, kernel, iterations=1)

            # Применяем маску к исходному кадру
            red_filtered_frame = cv.bitwise_and(frame, frame, mask=mask)

            # Находим контуры на маске
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Находим моменты первого порядка для первого контура
                moments = cv.moments(contours[0])

                # Вычисляем площадь объекта
                dM01 = moments['m01']
                dM10 = moments['m10']
                dArea = moments['m00']

                if dArea > 100:
                    x = int(dM10 / dArea)
                    y = int(dM01 / dArea)

                    max_cnt = cv.contourArea(contours[0])
                    max_cnt_obj = contours[0]
                    for cnt in contours:
                        if max_cnt < cv.contourArea(cnt): 
                            max_cnt = cv.contourArea(cnt)
                            max_cnt_obj = cnt
                    x, y, w, h = cv.boundingRect(max_cnt_obj)

                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 5)

            cv.imshow('hsv', red_filtered_frame)
            cv.imshow('cap', frame)
            cv.imshow('mask', mask)

            if cv.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()


Test.class_work()
