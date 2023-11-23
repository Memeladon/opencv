import cv2 as cv


def motion_detection(src, output, ksize, sigma, threshold, base_area):
    # Читаем видео из файла
    cap = cv.VideoCapture(src, cv.CAP_ANY)

    ret, frame = cap.read()
    if not cap.isOpened():
        print("Невозможно прочесть видео.")
        exit()

    # Бинаризация изображения
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Гауссовское размытие
    gg_image = cv.GaussianBlur(gray_image, (ksize, ksize), sigma)
    # Кодек
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    # Высота и ширина кадра для записи
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv.CAP_PROP_FPS)

    video_writer = cv.VideoWriter('Output_videos/' + output + '.mov', fourcc, fps, (width, height))
    video_writer_contours = cv.VideoWriter('Output_videos/' + output + '_countrs.mov', fourcc, fps, (width, height))

    while True:

        # Copy предыдущий кадр
        prev_image = gg_image.copy()

        # Новый кадр
        ret, next_frame = cap.read()

        if not ret:
            break

        gray_image = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
        gg_image = cv.GaussianBlur(gray_image, (ksize, ksize), sigma)

        # Вычисляет абсолютную разницу между двумя изображениями.
        frame_diff = cv.absdiff(gg_image, prev_image)

        # Применяет пороговое(бинарное) значение к изображению. Тип пороговой обработки
        _, frame_treshold = cv.threshold(frame_diff, threshold, 255, cv.THRESH_BINARY)

        # Поиск контуров. Режим поиска контура. Метод аппроксимации контура.
        contours, hierarchy = cv.findContours(frame_treshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Площадь контура
            area = cv.contourArea(contour)

            if area > base_area:
                video_writer.write(next_frame)
                countrs_frame = next_frame.copy()
                cv.drawContours(countrs_frame, [contour], 0, (0, 255, 0), 2)
                video_writer_contours.write(countrs_frame)

    video_writer.release()
    video_writer_contours.release()


# Размытие
ksize = 7
sigma = 2
# Обнаружение ложных срабатываний
threshold = 120
base_area = 50
motion_detection('Videos/LR4_main_video.mov', 'final_video_1', ksize, sigma, threshold, base_area)

# Размытие
ksize = 5
sigma = 0
# Обнаружение ложных срабатываний
threshold = 30
base_area = 10
motion_detection('Videos/LR4_main_video.mov', 'final_video_2', ksize, sigma, threshold, base_area)

# Размытие
ksize = 9
sigma = 2
# Обнаружение ложных срабатываний
threshold = 35
base_area = 8
motion_detection('Videos/LR4_main_video.mov', 'final_video_3', ksize, sigma, threshold, base_area)

# Размытие
ksize = 7
sigma = 1.5
# Обнаружение ложных срабатываний
threshold = 45
base_area = 6
motion_detection('Videos/LR4_main_video.mov', 'final_video_4', ksize, sigma, threshold, base_area)
