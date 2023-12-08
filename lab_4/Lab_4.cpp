#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>

#define _USE_MATH_DEFINES

#define  M_PI  3.1415926535897932384626433

using namespace cv;
using namespace std;
class MyGaussianBlur {
public:
    static vector<vector<float>> gaussian_filter(pair<int, int> filter_shape, float sigma) {
        int m = filter_shape.first;
        int n = filter_shape.second;
        int m_half = m / 2;
        int n_half = n / 2;

        vector<vector<float>> gaussian_filter(m, vector<float>(n, 0.0));
        float sum_of_el = 0.0;

        for (int y = -m_half; y <= m_half; y++) {
            for (int x = -n_half; x <= n_half; x++) {
                float first_part = 1.0 / (2.0 * M_PI * sigma * sigma);
                float exp_term = exp(-(x * x + y * y) / (2.0 * sigma * sigma));
                gaussian_filter[y + m_half][x + n_half] = first_part * exp_term;
                sum_of_el += first_part * exp_term;
            }
        }

        for (int y = -m_half; y <= m_half; y++) {
            for (int x = -n_half; x <= n_half; x++) {
                gaussian_filter[y + m_half][x + n_half] /= sum_of_el;
            }
        }

        return gaussian_filter;
    }

    static Mat convolution(Mat image, vector<vector<float>> kernel) {
        int channels = image.channels();
        int nRows = image.rows;
        int nCols = image.cols;

        int kernel_size = kernel[0].size();

        int result_height = nRows - kernel_size + 1;
        int result_width = nCols - kernel_size + 1;

        Mat result;
        result = Mat(image.size(), CV_8U);

        double min_sum = DBL_MAX;
        double max_sum = -DBL_MAX;

        if (channels == 1)
        {
            for (int i = 0; i < result_height; i++) {
                for (int j = 0; j < result_width; j++) {
                    double sum = 0.0;

                    for (int n = 0; n < kernel_size; n++) {
                        for (int m = 0; m < kernel_size; m++) {
                            sum += image.at<uchar>(i + n, j + m) * kernel[n][m];
                        }
                    }

                    if (sum < min_sum) {
                        min_sum = sum;
                    }
                    if (sum > max_sum) {
                        max_sum = sum;
                    }
                }
            }

            for (int i = 0; i < result_height; i++) {
                for (int j = 0; j < result_width; j++) {
                    double sum = 0.0;

                    for (int n = 0; n < kernel_size; n++) {
                        for (int m = 0; m < kernel_size; m++) {
                            sum += image.at<uchar>(i + n, j + m) * kernel[n][m];
                        }
                    }
                    double normalized_sum = (sum - min_sum) * (255.0 / (max_sum - min_sum));
                    result.at<uchar>(i + kernel_size / 2, j + kernel_size / 2) = (normalized_sum);
                }
            }

            return result;
        }
        else {
            result = Mat(image.size(), CV_8U);
            for (int i = 0; i < result_height; i++) {
                for (int j = 0; j < result_width; j++) {

                    for (int m = 0; m < kernel_size; m++) {
                        for (int n = 0; n < kernel_size; n++) {
                            for (int g = 0; g < kernel_size; g++) {
                                Vec3b pixel = image.at<Vec3b>(i + kernel_size / 2, j + kernel_size / 2);

                                Vec3b pixelSd = image.at<Vec3b>(i + m, j + g);
                                pixel[0] += pixelSd[0] * kernel[g][n];
                                pixel[1] += pixelSd[1] * kernel[g][n];
                                pixel[2] += pixelSd[2] * kernel[g][n];
                                result.at<Vec3b>(i + kernel_size / 2, j + kernel_size / 2) = pixel;
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
};
class MyKanni {
public:
    static void start(std::string path, pair<int, int> filter_shape, double sigma, double low_pr, double high_pr) {
        Mat img = prepare_img(path, filter_shape, sigma);
        imshow("Gray Image", img);
        int k2 = waitKey(0);
        Mat gradient_magnitude, gradient_angle;
        search_for_gradients(img, gradient_magnitude, gradient_angle);
        Mat suppressed = non_max_suppression(gradient_magnitude, gradient_angle);
        Mat result_of_method = double_threshold_filtering(suppressed, low_pr, high_pr);

        imshow("Result Image", result_of_method);
        int t = waitKey(0);
    }

    static Mat prepare_img(std::string path, pair<int, int> filter_shape, double sigma) {
        Mat    img = imread(path);
        cvtColor(img, img, COLOR_BGR2GRAY);

        vector<vector<float>> kernel = MyGaussianBlur::gaussian_filter(filter_shape, sigma);
        return MyGaussianBlur::convolution(img, kernel);
    }

    static vector<vector<float>> get_sobel_operator_x() {
        return { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    }

    static vector<vector<float>> get_sobel_operator_y() {
        return { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };
    }


    static void search_for_gradients(Mat img, Mat& gradient_magnitude, Mat& gradient_angle) {
        Mat gradient_x = MyGaussianBlur::convolution(img, get_sobel_operator_x());
        Mat gradient_y = MyGaussianBlur::convolution(img, get_sobel_operator_y());

        /*imshow("gradient_x", gradient_x);
        int k1 = waitKey(0);
        imshow("gradient_y", gradient_y);
        int k = waitKey(0);*/

        gradient_angle = Mat(gradient_x.size(), CV_32F);
        gradient_magnitude = Mat(gradient_x.size(), CV_8U);

        double min_sum = DBL_MAX;
        double max_sum = -DBL_MAX;
        for (int i = 0; i < gradient_x.rows; i++) {
            for (int j = 0; j < gradient_x.cols; j++) {
                double x = gradient_x.at<uchar>(i, j);
                double y = gradient_y.at<uchar>(i, j);
                double result = round((atan2(x, y)) / (M_PI / 4)) * (M_PI / 4) - (M_PI / 2);
                gradient_angle.at<float>(i, j) = result;

                double sum = sqrt(x * x + y * y);
                if (sum < min_sum)
                    min_sum = sum;
                if (sum > max_sum)
                    max_sum = sum;
            }
        }
        for (int i = 0; i < gradient_x.rows; i++) {
            for (int j = 0; j < gradient_x.cols; j++) {
                double x = gradient_x.at<uchar>(i, j);
                double y = gradient_y.at<uchar>(i, j);
                double sum = sqrt(x * x + y * y);
                double normalized_sum = (sum - min_sum) * (255.0 / (max_sum - min_sum));
                gradient_magnitude.at<uchar>(i, j) = (normalized_sum);
            }
        }

        imshow("Gradient Value", gradient_magnitude);
        int k3 = waitKey(0);
    }

    static int classify_gradient(uchar gradient_x, uchar gradient_y, double tg) {
        if ((gradient_x > 0 && gradient_y < 0 && tg < -2.414) || (gradient_x < 0 && gradient_y < 0 && tg > 2.414)) {
            return 0;
        }
        else if (gradient_x > 0 && gradient_y < 0 && tg < -0.414) {
            return 1;
        }
        else if ((gradient_x > 0 && gradient_y < 0 && tg > -0.414) || (gradient_x > 0 && gradient_y > 0 && tg < 0.414)) {
            return 2;
        }
        else if (gradient_x > 0 && gradient_y > 0 && tg < 2.414) {
            return 3;
        }
        else if ((gradient_x > 0 && gradient_y > 0 && tg > 2.414) || (gradient_x < 0 && gradient_y > 0 && tg < -2.414)) {
            return 4;
        }
        else if (gradient_x < 0 && gradient_y > 0 && tg < -0.414) {
            return 5;
        }
        else if ((gradient_x < 0 && gradient_y > 0 && tg > -0.414) || (gradient_x < 0 && gradient_y < 0 && tg < 0.414)) {
            return 6;
        }
        else if (gradient_x < 0 && gradient_y < 0 && tg < 2.414) {
            return 7;
        }
        else {
            return -1;
        }
    }

    static Mat non_max_suppression(Mat gradient_magnitude, Mat gradient_angle) {
        Mat suppressed = gradient_magnitude.clone();

        gradient_angle *= 180.0 / CV_PI;
        gradient_angle = Mat(gradient_angle.size(), CV_32F);

        for (int i = 1; i < gradient_magnitude.rows - 1; i++) {
            for (int j = 1; j < gradient_magnitude.cols - 1; j++) {
                float angle = gradient_angle.at<float>(i, j);

                vector<uchar> neighbors;

                if ((0 <= angle < 22.5) || (157.5 <= angle <= 180)) {
                    neighbors = { gradient_magnitude.at<uchar>(i, j - 1), gradient_magnitude.at<uchar>(i, j + 1) };
                }
                else if (22.5 <= angle < 67.5) {
                    neighbors = { gradient_magnitude.at<uchar>(i - 1, j - 1), gradient_magnitude.at<uchar>(i + 1, j + 1) };
                }
                else if (67.5 <= angle < 112.5) {
                    neighbors = { gradient_magnitude.at<uchar>(i - 1, j), gradient_magnitude.at<uchar>(i + 1, j) };
                }
                else {
                    neighbors = { gradient_magnitude.at<uchar>(i - 1, j + 1), gradient_magnitude.at<uchar>(i + 1, j - 1) };
                }

                if (gradient_magnitude.at<uchar>(i, j) < *std::max_element(neighbors.begin(), neighbors.end())) {
                    suppressed.at<uchar>(i, j) = 0;
                }
            }
        }

        return suppressed;
    }

    static Mat double_threshold_filtering(Mat img, double low_pr, double high_pr) {
        int down = low_pr * 255;
        int up = high_pr * 255;

        Mat result = img.clone();
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                if (img.at<uchar>(i, j) >= up) {
                    result.at<uchar>(i, j) = 255;
                }
                else if (img.at<uchar>(i, j) <= down) {
                    result.at<uchar>(i, j) = 0;
                }
                else {
                    result.at<uchar>(i, j) = 127;
                }
            }
        }

        return result;
    }

};

int main() {
    string image_path = "img.png";
    Mat image = imread(image_path);
    imshow("Host Image", image);
    int e = waitKey(0);
    pair<int, int> filter_shape(3, 3);

    MyKanni::start(image_path, filter_shape, 7, 0.5, 1);

    return 0;
}