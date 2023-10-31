import numpy as np
import cv2

class KCFTracker:
    def __init__(self, image, bbox):
        self.x, self.y, self.width, self.height = bbox
        self.sigma = 0.2
        self.cell_size = 4
        self.interp_factor = 0.075
        self.target = np.zeros((bbox[3], bbox[2]), dtype=np.float32)
        self.padding = 1.5
        self.init_features(image)

    def gaussian_correlation(self, x1, x2):
        x1 = x1[:self.features.shape[0], :self.features.shape[1]]
        x2 = x2[:self.features.shape[0], :self.features.shape[1]]

        c = np.fft.ifft2(np.fft.fft2(x1) * np.conj(np.fft.fft2(x2)))
        c = np.fft.fftshift(c)
        c /= np.max(np.abs(c))
        return np.real(np.fft.ifft2(c))

    def create_gaussian_peak(self, size):
        y, x = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * self.sigma ** 2)))
        return g / np.max(g)

    def init_features(self, image):
        size = int(self.padding * self.cell_size)
        self.features = self.create_gaussian_peak(size)
        self.features = np.fft.fft2(self.features)

        x1 = int(max(0, self.x - self.width * self.padding / 2))
        y1 = int(max(0, self.y - self.height * self.padding / 2))
        x2 = int(min(image.shape[1], self.x + self.width * self.padding / 2))
        y2 = int(min(image.shape[0], self.y + self.height * self.padding / 2))
        patch = image[y1:y2, x1:x2]
        gray_patch_resized = cv2.resize(patch, (self.target.shape[1], self.target.shape[0]))
        self.target = np.fft.fft2(gray_patch_resized)

    def detect(self, x):
        k = self.gaussian_correlation(x, self.target)
        response = np.real(np.fft.ifft2(np.fft.fft2(k) * np.fft.fft2(self.features)))
        response = np.fft.fftshift(response)
        max_response = response.max()
        max_loc = np.unravel_index(response.argmax(), response.shape)
        self.x += max_loc[1] - response.shape[1] // 2
        self.y += max_loc[0] - response.shape[0] // 2

    def update(self, patch, image):
        x1 = int(max(0, self.x - self.width * self.padding / 2))
        y1 = int(max(0, self.y - self.height * self.padding / 2))
        x2 = int(min(image.shape[1], self.x + self.width * self.padding / 2))
        y2 = int(min(image.shape[0], self.y + self.height * self.padding / 2))
        patch = image[y1:y2, x1:x2]
        gray_patch_resized = cv2.resize(patch, (self.target.shape[1], self.target.shape[0]))
        gray_patch_resized = np.fft.fft2(gray_patch_resized)
        self.detect(gray_patch_resized)
        self.target = (1 - self.interp_factor) * self.target + self.interp_factor * gray_patch_resized

    def track(self, image):
        x1 = int(max(0, self.x - self.width * self.padding / 2))
        y1 = int(max(0, self.y - self.height * self.padding / 2))
        x2 = int(min(image.shape[1], self.x + self.width * self.padding / 2))
        y2 = int(min(image.shape[0], self.y + self.height * self.padding / 2))
        patch = image[y1:y2, x1:x2]

        # Выполните обновление перед детекцией
        self.update(patch, image)

        # Теперь выполните детекцию на обновленном объекте
        self.detect(np.fft.fft2(cv2.resize(patch, (self.target.shape[1], self.target.shape[0]))))

        # Обновите текущее положение объекта
        self.x = x1 + (x2 - x1) / 2
        self.y = y1 + (y2 - y1) / 2



video = cv2.VideoCapture('data_files/horse_lunch.webm')
ret, frame = video.read()

bbox = cv2.selectROI(frame, False)

tracker = KCFTracker(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), bbox)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.track(gray_frame)

    x, y, width, height = tracker.x, tracker.y, tracker.width, tracker.height
    x1 = max(0, int(x - width * tracker.padding / 2))
    y1 = max(0, int(y - height * tracker.padding / 2))
    x2 = min(frame.shape[1], int(x + width * tracker.padding / 2))
    y2 = min(frame.shape[0], int(y + height * tracker.padding / 2))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Tracking', frame)

    if cv2.waitKey(50) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
