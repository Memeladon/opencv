import cv2
import numpy as np
import time

def apply_canny(image, operator, gaussian_blur_kernel, low_threshold, high_threshold):
    # Преобразование в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение размытия Гаусса
    blurred_image = cv2.GaussianBlur(gray_image, (gaussian_blur_kernel, gaussian_blur_kernel), 0)

    # Замер времени выполнения
    start_time = time.time()

    # Применение оператора границ (Канни, Лаплас, Робертса, Превитта или альтернативного метода)
    if operator == 'canny':
        edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    elif operator == 'laplacian':
        edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
    elif operator == 'prewitt':
        # Применение альтернативного оператора (например, оператор Превитта)
        prewitt_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        prewitt_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(prewitt_x**2 + prewitt_y**2).astype(np.uint8)
    elif operator == 'roberts':
        # Применение оператора Робертса
        roberts_cross_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_cross_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gradient_x = cv2.filter2D(blurred_image, cv2.CV_64F, roberts_cross_x)
        gradient_y = cv2.filter2D(blurred_image, cv2.CV_64F, roberts_cross_y)
        edges = np.sqrt(gradient_x**2 + gradient_y**2).astype(np.uint8)
    else:
        return None  # Вернуть None в случае неверного оператора

    # Завершение замера времени выполнения
    elapsed_time = time.time() - start_time
    print(f"Execution time ({operator}): {elapsed_time:.4f} seconds")

    return edges, elapsed_time


def test_canny_parameters(image_paths, operator):
    for path in image_paths:
        # Загрузка изображения
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        # Тестирование разных параметров размытия Гаусса
        average_execution_time = 0.0
        num_tests = 0

        for blur_kernel in [3, 5, 7]:
            # Применение алгоритма Канни, Лапласа, Робертса, Превитта или альтернативного метода с разными параметрами
            for low_threshold, high_threshold in [(50, 150), (100, 200), (150, 250)]:
                edges, execution_time = apply_canny(image, operator, blur_kernel, low_threshold, high_threshold)

                if edges is not None:
                    # Визуализация результатов
                    cv2.imshow(f"Parameters: Operator={operator}, Blur={blur_kernel}, Low={low_threshold}, High={high_threshold}",
                               edges)
                    cv2.waitKey(0)

                    average_execution_time += execution_time
                    num_tests += 1

        # Вывод среднего времени выполнения для одной картинки
        if num_tests > 0:
            average_execution_time /= num_tests
            print(f"Average execution time for {operator} on image {path}: {average_execution_time:.4f} seconds")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Замените пути к изображениям на свои
    image_paths = ["11.jpg", "7.jpg", "7.jpg", "8.jpg","9.jpg","10.jpg"]

    # Выбор между алгоритмом Канни, оператором Лапласа, оператором Робертса, оператором Превитта и альтернативным методом
    # test_canny_parameters(image_paths, operator='canny')
    # test_canny_parameters(image_paths, operator='laplacian')
    # test_canny_parameters(image_paths, operator='roberts')
    test_canny_parameters(image_paths, operator='prewitt')
