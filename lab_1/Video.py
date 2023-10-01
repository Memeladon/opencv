import cv2


def video_render(video):
    cap = cv2.VideoCapture(video, cv2.CAP_ANY)

    # cv2.namedWindow ('Video', cv2.WINDOW_FREERATIO)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.9, fy=1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break


video_render('data/times-square-in-new-york-city-4k.mp4')
