import cv2
import numpy as np

from lab_4.misc import sobel_filter, non_max_suppression, double_threshold_filtering


def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 2)
    return blurred_image


def display_image(image, winname):
    cv2.imshow(f"{winname}", image)


def view(image_path):
    try:
        image = cv2.imread(image_path)
        if image is not None:
            if len(image.shape) > 2 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            max_size = max(image.shape)
            if max_size > 700:
                scale = 700 / max_size
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            blurred_image = apply_gaussian_blur(image)

            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            # Применяем операторы Собеля для вычисления градиентов
            gradient_x = sobel_filter(image, kernel_x)
            gradient_y = sobel_filter(image, kernel_y)

            # Вычисляем матрицу значений длин градиентов
            gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

            # Вычисляем матрицу значений углов градиентов
            gradient_angle = np.arctan2(gradient_y, gradient_x)
            # Подавление немаксимумов
            non_max = non_max_suppression(gradient_magnitude, gradient_angle)

            # Двойнуя пороговая фильтрация
            dbl_thr_filtered = double_threshold_filtering(non_max, 0.2, 0.9)

            display_image(blurred_image, 'Grey')
            display_image(dbl_thr_filtered, 'Double Threshold Filtering')
            cv2.waitKey(0)
        else:
            print("Failed to read the image.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    view('../data/cat_wat.jpg')
