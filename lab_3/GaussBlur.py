from lab_3.GaussMatrix import *


def webcam_to_grayscale():
    # Открываем видеокамеру
    cap = cv2.VideoCapture(0)

    while True:
        # Считываем кадр с камеры
        ret, frame = cap.read()

        if not ret:
            break

        # Переводим кадр в чб режим
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Применение фильтра Гаусса с разными параметрами
        sigma = 2.0
        kernel_size = (5, 5)
        blurred_frame_opencv = cv2.GaussianBlur(grayscale_frame, kernel_size, sigma)
        blurred_frame = apply_gaussian_filter(grayscale_frame, 5, 2.0)

        # Отображаем кадр
        cv2.imshow('Original Image', grayscale_frame)
        cv2.imshow('Blurred Image (OpenCV)', blurred_frame_opencv)
        cv2.imshow('Blurred Image (Custom)', blurred_frame)

        # Для выхода из цикла нажмите клавишу 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Освобождаем ресурсы и закрываем окна OpenCV
    cap.release()
    cv2.destroyAllWindows()


# Вызываем функцию для работы с веб-камерой в чб режиме
webcam_to_grayscale()
