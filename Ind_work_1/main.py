import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    # На horse_racing1 работает корректно CSRT - запусти на всех чекни
    # Все переключаются кста на перегоняющего(выделяй первого коня)
    tracker_types = ['MIL', 'KCF', 'CSRT']
    tracker_type = tracker_types[0]

    # Создаём объект трекера выбранного класса
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:

        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()

    video = cv2.VideoCapture("data_files/Глеб1.avi")

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Выбираем кодек FFV1
    out = cv2.VideoWriter('output_data/Глеб1_head_MIL.avi', fourcc, 30, (int(video.get(3)), int(video.get(4))))

    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))

    # Преобразуем числовой код формата в строку
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print(f"Формат видео: {fourcc_str}")

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Прочитали первый кадр
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Шляпу выделяем(выбираем область для инициализации в виде прямоугольника)
    bbox = cv2.selectROI(frame, False)

    # Выведет список с 4-мя числами (первые 2 - коорды верхней точки следующие - размеры прямоугольника)
    print(bbox)

    # Проверка на успешность инициализации трекера
    # И инициализация (Это необходимо для того, чтобы начать отслеживать объект с указанной позиции на первом кадре)
    ok = tracker.init(frame, bbox)

    while True:

        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # На основании предыдущего состояния объекта(tracker) и читаемого фрейма(frame) пытаемся определить состояние объекта
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Если читается, рисуем прямоугольник
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        out.write(frame)

        # Тип используемого трекера
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Кадры в секунду
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # объединили всю шляпу
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(50) & 0xff
        if k == 27: break
