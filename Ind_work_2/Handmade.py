import cv2
import numpy as np

def find_contours(binary_mask):
    # Создаем пустую матрицу для хранения результата
    result = np.zeros_like(binary_mask)

    # Проходим по каждому пикселю внутри изображения
    for i in range(1, binary_mask.shape[0] - 1):
        for j in range(1, binary_mask.shape[1] - 1):
            # Если текущий пиксель - белый
            if binary_mask[i, j] == 255:
                # Проверяем вокруг текущего пикселя
                neighbors = binary_mask[i - 1:i + 2, j - 1:j + 2].flatten()

                # Если хотя бы один из соседей - черный, то текущий пиксель находится на границе
                if 0 in neighbors:
                    result[i, j] = 255

    return result

def main():
    # Загрузка изображения
    image = cv2.imread('in/car_1.jpg')

    # Преобразование изображения из BGR в HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определение диапазона цвета
    lower_purple = np.array([110, 50, 50])
    upper_purple = np.array([160, 255, 255])

    # Создание маски с использованием цветового фильтра
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Применение операций морфологии для улучшения результатов
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours = find_contours(mask)

    # Отображение исходного и обработанного изображений
    cv2.imshow('Original Image', image)
    cv2.imshow('Contours', contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
