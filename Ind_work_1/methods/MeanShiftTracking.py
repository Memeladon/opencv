import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Инициализируем каскадный классификатор для обнаружения лиц.
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
# Ищем лица на первом кадре.
face_rects = face_cascade.detectMultiScale(frame)
# Извлекаем координаты и размер первого обнаруженного лица.
(face_x, face_y, w, h) = tuple(face_rects[0])

# Инициализируем окно отслеживания для среднего сдвига (mean shift).
track_window = (face_x, face_y, w, h)

# Выделяем область интереса (Region of Interest, ROI) на первом кадре, где находится лицо.
roi = frame[face_y:face_y + h, face_x:face_x + w]
# Конвертируем ROI в цветовое пространство HSV.
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Вычисляем гистограмму цветового канала H (оттенок) для ROI.
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

# Нормализуем гистограмму для лучшей совместимости с методом mean shift.
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Инициализируем критерий завершения mean shift алгоритма.
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Вычисляем обратное проецирование (back projection) с использованием гистограммы ROI.
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Применяем mean shift для обновления координат окна отслеживания.
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Извлекаем новые координаты и размер окна отслеживания.
        x, y, w, h = track_window

        # Рисуем прямоугольник вокруг обнаруженного лица на кадре.
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

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
