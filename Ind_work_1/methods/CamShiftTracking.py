import cv2
import numpy as np
import os
import time


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

        output_folder = f'../videos/CamShift_data(built-in)'
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f'CamShift_1.avi')
        self.output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 60.0,
                                      (self.frame_width, self.frame_height), True)

        # Изменение размера видео для более удобного просмотра
        self.frame = cv2.resize(self.frame, (self.frame_width , self.frame_height))

    def track(self):
        # Инициализируем каскадный классификатор для обнаружения лиц.
        face_cascade = cv2.CascadeClassifier('../facedetecting_cascade.xml')

        # Ищем лица на первом кадре.
        face_rects = face_cascade.detectMultiScale(self.frame)

        # Извлекаем координаты и размер первого обнаруженного лица.
        (face_x, face_y, w, h) = tuple(face_rects[0])

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

        prev_frame_time = 0

        # Замер времени выполнения
        start_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading video file.")
                break

            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Вычисляем обратное проецирование (back projection) с использованием гистограммы ROI.
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            x0, y0, x1, y1 = track_window

            # Применяем метод CamShift для обновления координат окна отслеживания.
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            x, y, w, h = track_window

            # Рисуем прямоугольник, охватывающий область отслеживания, на кадре.
            # img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
            img = cv2.ellipse(frame, ret, (0, 0, 255), 2)

            # Для примера, вывод FPS на изображение
            font = cv2.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))

            cv2.putText(frame, fps, (3, 30), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

            # Отображаем текущий кадр с обнаруженным лицом.
            cv2.imshow('img', img)
            self.output.write(frame)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        # Закрываем окна OpenCV и освобождаем видеопоток.
        cv2.destroyAllWindows()
        self.cap.release()
        # Завершение замера времени выполнения
        elapsed_time = time.time() - start_time
        print(f"Execution time (camshift): {elapsed_time:.4f} seconds")

if __name__ == '__main__':
    tracker = CamShiftHandMade()
    video_path = '../videos/source/5.mp4'
    tracker.video_choose(video_path)
    tracker.track()
