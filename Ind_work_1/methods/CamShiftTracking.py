import cv2
import numpy as np


class CamShiftHandMade:
    def __init__(self):
        self.frame_height = 0
        self.frame_width = 0
        self.frame = None
        self.ret = False

    def video_choose(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()

        if self.ret:
            self.frame_height, self.frame_width = self.frame.shape[:2]
            print(self.frame_width, self.frame_height)
        else:
            print("The video was not read successfully")

        # Изменение размера видео для более удобного просмотра
        self.frame = cv2.resize(self.frame, (self.frame_width // 2, self.frame_height // 2))

    def track(self):
        # Инициализируем каскадный классификатор для обнаружения лиц.
        # face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
        #
        # # Ищем лица на первом кадре.
        # face_rects = face_cascade.detectMultiScale(self.frame)
        #
        # if len(face_rects) > 0:
        bbox = cv2.selectROI(self.frame, False)

        # Извлекаем координаты и размер первого обнаруженного лица.
        face_x, face_y, w, h = bbox

        # Инициализируем окно отслеживания для метода CamShift.
        track_window = (face_x, face_y, w, h)

        # Выделяем область интереса (Region of Interest, ROI) на первом кадре, где находится лицо.
        roi = self.frame[face_y:face_y + h, face_x:face_x + w]
        # Конвертируем ROI в цветовое пространство HSV.
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Вычисляем гистограмму цветового канала H (оттенок) для ROI.
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        # Нормализуем гистограмму для лучшей совместимости с методом CamShift.
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Инициализируем критерий завершения для метода CamShift.
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading video file.")
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Вычисляем обратное проецирование (back projection) с использованием гистограммы ROI.
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            x0, y0, x1, y1 = track_window

            # Применяем метод CamShift для обновления координат окна отслеживания.
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            x, y, w, h = track_window

            # Рисуем прямоугольник, охватывающий область отслеживания, на кадре.
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

            # Отображаем текущий кадр с обнаруженным лицом.
            cv2.imshow('img', img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        # Закрываем окна OpenCV и освобождаем видеопоток.
        cv2.destroyAllWindows()
        self.cap.release()


if __name__ == '__main__':
    tracker = CamShiftHandMade()
    video_path = '../videos/source/1.mp4'
    tracker.video_choose(video_path)
    tracker.track()
