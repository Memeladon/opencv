import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Инициализируем каскадный классификатор для обнаружения лиц.
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

# Ищем лица на первом кадре.
face_rects = face_cascade.detectMultiScale(frame)

# Извлекаем координаты и размер первого обнаруженного лица.
(face_x, face_y, w, h) = tuple(face_rects[0])

# Инициализируем окно отслеживания для метода CamShift.
track_window = (face_x, face_y, w, h)

# Выделяем область интереса (Region of Interest, ROI) на первом кадре, где находится лицо.
roi = frame[face_y:face_y + h, face_x:face_x + w]
# Конвертируем ROI в цветовое пространство HSV.
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Вычисляем гистограмму цветового канала H (оттенок) для ROI.
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
# Нормализуем гистограмму для лучшей совместимости с методом CamShift.
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Инициализируем критерий завершения для метода CamShift.
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    # недоделал
    ret, frame = cap.read()
    vis = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    if ret:

        # Вычисляем обратное проецирование (back projection) с использованием гистограммы ROI.
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        x0, y0, x1, y1 = track_window
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        # Применяем метод CamShift для обновления координат окна отслеживания.
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Извлекаем угол поворота и координаты вершин области отслеживания.
        pts = cv2.boxPoints(ret)
        pts = np.intp(pts)

        # Рисуем многоугольник, охватывающий область отслеживания, на кадре.
        img = cv2.polylines(frame, [pts], True, (0, 0, 255), 5)

        # Отображаем текущий кадр с обнаруженным лицом.
        cv2.imshow('img', img)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    else:
        break

# Закрываем окна OpenCV и освобождаем видеокамеру.
cv2.destroyAllWindows()
cap.release()
