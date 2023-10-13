import cv2
import numpy as np

'''
size: Это размерность ядра Гаусса, которая указывает, сколько строк и столбцов будет в создаваемой матрице ядра.
sigma: Это параметр, который определяет степень размытия. 
Чем больше значение sigma, тем более размыто будет изображение.
'''


# Функция служит для создания такого Гауссовского ядра, которое затем будет использоваться для размытия изображения
def create_gaussian_kernel(size, sigma):
    # Cоздает матрицу на основе указанной функции.
    # Функция имеет два аргумента x и y, которые представляют координаты пикселей в создаваемой матрице
    # G(x, y) = (1 / (2 * π * σ ^ 2)) * exp(-((x - μ) ^ 2 + (y - ν) ^ 2) / (2 * σ ^ 2))

    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel


# Нормируется, чтобы сумма всех ее элементов составляла 1.
def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


def custom_gaussian_filter_old(image, kernel_size, sigma):
    # Создание матрицы Гаусса
    kernel = create_gaussian_kernel(kernel_size, sigma)
    normalized_kernel = normalize_kernel(kernel)

    # Применение фильтра
    filtered_image = cv2.filter2D(image, -1, normalized_kernel)

    return filtered_image


def apply_gaussian_filter(frame, kernel_size, sigma):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    normalized_kernel = normalize_kernel(kernel)
    filtered_image = cv2.filter2D(frame, -1, normalized_kernel)
    return filtered_image


def print_matrix():
    # Пример матрицы Гаусса размерности 5x5
    gaussian_kernel = create_gaussian_kernel(5, 1.0)
    print(f"Гауссово ядро (size {5}x{5}):")
    print(gaussian_kernel)

    # Пример нормирования матрицы Гаусса размерности 5x5
    normalized_kernel = normalize_kernel(gaussian_kernel)
    print("Нормализованное Гауссовское ядро (size 5x5):")
    print(normalized_kernel)
