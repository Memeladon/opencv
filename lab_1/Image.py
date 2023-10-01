import cv2


def viewImage(image):
    img1 = cv2.imread(image)
    img2 = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(image, cv2.IMREAD_LOAD_GDAL)

    cv2.namedWindow('Image1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Image2', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('Image3', cv2.WINDOW_AUTOSIZE)

    cv2.imshow('Image1', img1)
    cv2.imshow('Image2', img2)
    cv2.imshow('Image3', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
