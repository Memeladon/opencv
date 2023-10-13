import cv2
import numpy as np

# Открываем видеокамеру (0 - индекс камеры, может быть разным в зависимости от вашего оборудования)
cap = cv2.VideoCapture(0)

while True:
    # Считываем кадр с камеры
    ret, frame = cap.read()

    if not ret:
        break

    # Переводим кадр в формат HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Задаем диапазон красного цвета в формате HSV
    lower_red = np.array([0, 50, 10])  # Нижний порог для красного
    upper_red = np.array([10, 255, 255])  # Верхний порог для красного

    # Применяем фильтр для выделения красного цвета
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Применяем морфологическое преобразование для улучшения результата
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Открытие
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Закрытие

    # Находим контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Находим моменты первого порядка для первого контура
        moments = cv2.moments(contours[0])

        # Вычисляем площадь объекта
        area = moments['m00']

        if area > 0:
            # Находим координаты центра объекта
            cx = int(moments['m10'] / area)
            cy = int(moments['m01'] / area)

            # Рисуем черный прямоугольник вокруг объекта
            cv2.rectangle(frame, (cx - 50, cy - 50), (cx + 50, cy + 50), (0, 0, 0), 2)

    # Отображаем исходный кадр
    cv2.imshow('Video', frame)

    # Для выхода из цикла нажмите клавишу 'q'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Освобождаем ресурсы и закрываем окна OpenCV
cap.release()
cv2.destroyAllWindows()
