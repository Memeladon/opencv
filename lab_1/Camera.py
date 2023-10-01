import cv2


def readIPWriteTOFile():
    video = cv2.VideoCapture(0)
    ok, img = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    video_writer = cv2.VideoWriter("data/output.mov", fourcc, 25, (w, h))
    while True:
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def print_cam():
    # Создаем объект VideoCapture для чтения с камеры.
    cap = cv2.VideoCapture(0)  # 0 - индекс камеры по умолчанию

    # Устанавливаем размеры кадра (ширина и высота) на 640x480.
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        # Захватываем кадр с камеры.
        ret, frame = cap.read()

        # Преобразуем кадр в черно-белый.
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Отображаем кадр в окне с именем "frame".

        cv2.rectangle(frame, (310, 230), (330, 190), (0, 0, 255), 3)
        roi = frame[280:290 + 10, 330:360 + 10]
        # applying a gaussian blur over this new rectangle area
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        frame[280:290 + roi.shape[0], 330:360 + roi.shape[1]] = roi

        # frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
        # cv2.rectangle(frame, (310, 230), (330, 190), (0, 0, 255), 3)
        # cv2.rectangle(frame, (310, 290), (330, 250), (0, 0, 255), 3)
        # cv2.rectangle(frame, (280, 250), (360, 230), (0, 0, 255), 3)
        cv2.imshow('frame', frame)

        if not ret:
            print("Ошибка при захвате кадра.")
            break
        # Если нажата клавиша 'Esc' (код 27 в ASCII), то выходим из цикла.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Когда всё закончено, освобождаем ресурсы и закрываем окно.
    cap.release()
    cv2.destroyAllWindows()
