import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;

public class Main {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String imagePath = "5.jpg";
        int kernelSize = 5;
        double sigma = 2;

        Mat image = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);
        Imgproc.resize(image, image, new Size(image.width() / 2, image.height() / 2));

        Mat blurredImage = new Mat();
        Imgproc.GaussianBlur(image, blurredImage, new Size(kernelSize, kernelSize), sigma);

        Mat suppressedImage = nonMaxSuppression(blurredImage, kernelSize, sigma);

        double lowThreshold = 30;
        double highThreshold = 80;

        Mat cannyImage = doubleThresholding(suppressedImage, lowThreshold, highThreshold);

        Imgcodecs.imwrite("CannyEdgeDetected.jpg", cannyImage);
    }

    public static Mat nonMaxSuppression(Mat image, int kernelSize, double sigma) {
        Mat blurredImage = new Mat();
        Imgproc.GaussianBlur(image, blurredImage, new Size(kernelSize, kernelSize), sigma);

        Mat g_x = new Mat();
        Mat g_y = new Mat();
        Imgproc.Sobel(blurredImage, g_x, CvType.CV_64F, 1, 0, 3);
        Imgproc.Sobel(blurredImage, g_y, CvType.CV_64F, 0, 1, 3);

        Mat magnitude = new Mat();
        Core.magnitude(g_x, g_y, magnitude);

        Mat angle = new Mat();
        Core.phase(g_x, g_y, angle);

        Mat suppressedImage = Mat.zeros(magnitude.size(), magnitude.type());

        for (int i = 1; i < magnitude.rows() - 1; i++) {
            for (int j = 1; j < magnitude.cols() - 1; j++) {
                double mag = magnitude.get(i, j)[0];
                double theta = angle.get(i, j)[0] * 180 / Math.PI;

                double q = 255;
                double r = 255;

                if ((theta >= -22.5 && theta < 22.5) || (theta >= 157.5 && theta <= 180) || (theta < -157.5 && theta >= -180)) {
                    q = magnitude.get(i, j + 1)[0];
                    r = magnitude.get(i, j - 1)[0];
                } else if ((theta >= 22.5 && theta < 67.5) || (theta >= -157.5 && theta < -112.5)) {
                    q = magnitude.get(i - 1, j + 1)[0];
                    r = magnitude.get(i + 1, j - 1)[0];
                } else if ((theta >= 67.5 && theta < 112.5) || (theta >= -112.5 && theta < -67.5)) {
                    q = magnitude.get(i - 1, j)[0];
                    r = magnitude.get(i + 1, j)[0];
                } else if ((theta >= 112.5 && theta < 157.5) || (theta >= -67.5 && theta < -22.5)) {
                    q = magnitude.get(i - 1, j - 1)[0];
                    r = magnitude.get(i + 1, j + 1)[0];
                }

                if (mag >= q && mag >= r) {
                    suppressedImage.put(i, j, mag);
                }
            }
        }
        return suppressedImage;
    }

    public static Mat doubleThresholding(Mat image, double lowThreshold, double highThreshold) {
        Mat resultImage = new Mat(image.size(), image.type());

        for (int i = 1; i < image.rows() - 1; i++) {
            for (int j = 1; j < image.cols() - 1; j++) {
                double value = image.get(i, j)[0];
                if (value >= highThreshold) {
                    resultImage.put(i, j, 255);
                } else if (value >= lowThreshold && value < highThreshold) {
                    resultImage.put(i, j, 50);
                } else {
                    resultImage.put(i, j, 0);
                }
            }
        }
        return resultImage;
    }
}