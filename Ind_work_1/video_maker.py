import cv2

VIDEO_NAME = '"../videos/source/output_video.mp4"'


def record_video():
    # Инициализация видеозахвата для первой доступной камеры
    cap = cv2.VideoCapture(0)

    # Проверка, открылась ли камера
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    # Получение разрешения видеопотока
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создание объекта VideoWriter для записи видео в формате MP4
    # Формат видео кодека можно выбрать по желанию, но в данном случае используется XVID
    output_filename = VIDEO_NAME
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Проблема с захватом кадра")
            break

        # Запись кадра в видеофайл
        out.write(frame)

        # Отображение кадра в окне
        cv2.imshow('Video Recording', frame)

        # Нажмите клавишу 'q' для выхода из цикла записи видео
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    record_video()
