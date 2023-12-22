import cv2
import numpy as np


class CamShiftTracker(object):
    def __init__(self, curWindowRoi, imgBGR):
        # Инициализация объекта CAMShiftTracker с начальным окном и изображением
        self.updateCurrentWindow(curWindowRoi)
        self.updateHistograms(imgBGR)

        # Установка критериев завершения meanshift: 10 итераций или перемещение хотя бы на 1 пиксель
        self.term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def updateCurrentWindow(self, curWindowRoi):
        # Обновление координат текущего окна
        self.curWindow = curWindowRoi

    def updateHistograms(self, imgBGR):
        # Извлечение области интереса в BGR и преобразование в HSV
        self.bgrObjectRoi = imgBGR[self.curWindow[1]: self.curWindow[1] + self.curWindow[3],
                            self.curWindow[0]: self.curWindow[0] + self.curWindow[2]]
        self.hsvObjectRoi = cv2.cvtColor(self.bgrObjectRoi, cv2.COLOR_BGR2HSV)

        # Определение цветового диапазона в HSV и создание маски
        lower_range = np.array((0., 50., 50.))
        upper_range = np.array((180, 255., 255.))
        self.mask = cv2.inRange(self.hsvObjectRoi, lower_range, upper_range)

        # Вычисление гистограммы объекта в области интереса и ее нормализация
        hist_bins = 180
        self.histObjectRoi = cv2.calcHist([self.hsvObjectRoi], [0], self.mask, [hist_bins], [0, hist_bins])
        cv2.normalize(self.histObjectRoi, self.histObjectRoi, 0, 255, cv2.NORM_MINMAX)

    def getBackProjectedImage(self, imgBGR):
        # Преобразование входного изображения BGR в HSV
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        # Рассчет обратно-проекционного изображения с использованием гистограммы объекта
        backProjectedImg = cv2.calcBackProject([imgHSV], [0], self.histObjectRoi, [0, 180], 1)
        self.backProjectedImg = backProjectedImg
        return backProjectedImg.copy()

    def computeNewWindow(self, imgBGR):
        # Получение обратно-проекционного изображения
        self.getBackProjectedImage(imgBGR)
        # Использование CAMShift для вычисления нового окна для отслеживаемого объекта
        self.rotatedWindow, curWindow = cv2.CamShift(self.backProjectedImg, self.curWindow, self.term_criteria)
        # Преобразование повернутого окна в полигон
        self.rotatedWindow = cv2.boxPoints(self.rotatedWindow)
        self.rotatedWindow = np.int0(self.rotatedWindow)
        # Обновление текущего окна новыми координатами
        self.updateCurrentWindow(curWindow)

    def getCurWindow(self):
        # Возвращает координаты текущего окна
        return self.curWindow

    def getRotatedWindow(self):
        # Возвращает координаты повернутого окна
        return self.rotatedWindow


def execute_tracker(file):
    # Открытие видеофайла
    cap = cv2.VideoCapture(file)
    # Создание объекта VideoWriter для сохранения результата
    writer = cv2.VideoWriter(r"../videos/CAMSHIFT/1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 90,
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Инициализируем каскадный классификатор для обнаружения лиц.
    face_cascade = cv2.CascadeClassifier('../facedetecting_cascade.xml')

    # Чтение первого кадра
    ok, frame = cap.read()

    # Обнаружение лица с использованием каскада Хаара
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Используется Лицевой кадр как начальную область интереса (ROI)
    roi = (faces[0][0], faces[0][1], faces[0][2], faces[0][3])

    # Инициализация объекта CAMShiftTracker с выбранным ROI и первым кадром
    camShiftTracker = CamShiftTracker(roi, frame)

    while True:
        # Чтение следующего кадра
        ok, frame = cap.read()
        if not ok:
            break

        # Измерение времени для отслеживания
        timer = cv2.getTickCount()
        # Вычисление нового окна для отслеживаемого объекта
        camShiftTracker.computeNewWindow(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Получение координат повернутого окна
        rotatedWindow = camShiftTracker.getRotatedWindow()
        # Рисование полигона вокруг повернутого окна
        cv2.polylines(frame, [rotatedWindow], True, (0, 255, 0), 2, cv2.LINE_AA)

        # Отображение FPS на кадре
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Показать кадр с информацией об отслеживании
        cv2.imshow("Tracking an object using Camshaft", frame)
        # Запись кадра в выходное видео
        writer.write(frame)

        # Прерывание цикла при нажатии клавиши 'Esc'
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    # Освобождение объекта VideoWriter
    writer.release()


# Список видеофайлов для обработки
files = ['../videos/source/1.mp4',
         '../videos/source/2.mp4',
         '../videos/source/3.mp4',
         '../videos/source/4.mp4',
         '../videos/source/5.mp4']

# Цикл по списку файлов, вызов функции iz_part2 для каждого файла
for file in files:
    execute_tracker(file)
