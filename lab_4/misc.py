import numpy as np


def non_max_suppression(gradient_magnitude, gradient_angle):
    height, width = gradient_magnitude.shape
    suppressed = np.copy(gradient_magnitude)  # Создаем копию матрицы величин градиентов

    # Преобразуем углы градиента в диапазон [0, 180) градусов
    gradient_angle = (gradient_angle * 180.0 / np.pi) % 180

    for i in range(1, height - 1):  # Перебираем пиксели внутри изображения, исключая грани
        for j in range(1, width - 1):

            angle = gradient_angle[i, j]  # Угол градиента в текущем пикселе

            # Определение соседей в зависимости от угла градиента
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):  # Горизонтальное направление
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif 22.5 <= angle < 67.5:  # Диагональное направление
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
            elif 67.5 <= angle < 112.5:  # Вертикальное направление
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            else:  # Диагональное направление
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]

            # Если величина градиента в текущем пикселе меньше максимума среди соседей, устанавливаем ее в 0
            if gradient_magnitude[i, j] < max(neighbors):
                suppressed[i, j] = 0

    return suppressed  # Возвращаем изображение с подавленными немаксимумами


def double_threshold_filtering(img, low_pr, high_pr):
    # Преобразование пороговых значений в диапазон [0, 255]
    down = low_pr * 255
    up = high_pr * 255

    n, m = img.shape
    clone_of_img = np.copy(img)  # Создаем копию исходного изображения

    for i in range(n):
        for j in range(m):
            if clone_of_img[i, j] >= up:
                # Если значение пикселя выше верхнего порога, считаем его "сильной" границей (белый)
                clone_of_img[i, j] = 255
            elif clone_of_img[i, j] <= down:
                # Если значение пикселя ниже нижнего порога, считаем его фоном (черный)
                clone_of_img[i, j] = 0
            else:
                # Если значение пикселя между двумя порогами, считаем его "слабой" границей (серый)
                clone_of_img[i, j] = 127

    return clone_of_img  # Возвращаем изображение с примененным двойным порогом


def sobel_filter(img, kernel):
    try:
        img_height, img_width, img_canals = img.shape
    except:
        img_height, img_width = img.shape
        img_canals = 1

    kernel_height, kernel_width = kernel.shape

    # Вычисляем размеры результирующего изображения
    result_height = img_height - kernel_height + 1
    result_width = img_width - kernel_width + 1

    # Создаем пустой массив для результата с теми же размерами
    result = np.zeros((result_height, result_width), dtype=np.float32)

    if img_canals != 1:
        # Если изображение имеет несколько каналов (цветное изображение)
        for i in range(result_height):
            for j in range(result_width):
                for canal in range(img_canals):
                    # Применяем фильтр к каждому каналу
                    result[i, j, canal] = np.sum(img[i:i + kernel_height, j:j + kernel_width, canal] * kernel)
    else:
        # Если изображение одноканальное (оттенки серого)
        for i in range(result_height):
            for j in range(result_width):
                # Применяем фильтр к отдельному каналу
                result[i, j] = np.sum(img[i:i + kernel_height, j:j + kernel_width] * kernel)
    return result  # Возвращаем результат применения фильтра
