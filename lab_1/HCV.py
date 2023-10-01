import cv2


def HCV(img):
    frame = cv2.imread(img)

    cv2.namedWindow('Display window HCV', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('Display window no HCV', cv2.WINDOW_FREERATIO)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('Display window no HCV', frame)

    cv2.imshow('Display window HCV', hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
