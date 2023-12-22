import os
import time

import cv2


class TrackerCreator:
    def __init__(self, tracker_type, video_choose):
        super(TrackerCreator, self).__init__()

        self.tracker_type = tracker_type
        # Словарь доступных типов трекеров в OpenCV
        #  MIL, KCF и алгоритм TLD представляют собой предварительно обученные модели
        self.types_map = {'BOOSTING': cv2.legacy.TrackerBoosting_create(),
                          'MIL': cv2.TrackerMIL_create(),
                          'KCF': cv2.TrackerKCF_create(),
                          'TLD': cv2.legacy.TrackerTLD_create(),
                          'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create(),
                          'MOSSE': cv2.legacy.TrackerMOSSE_create(),
                          'CSRT': cv2.TrackerCSRT_create()}
        # Словарь доступных видео
        self.video_map = {'1': '../videos/source/1.mp4',
                          '2': '../videos/source/2.mp4',
                          '3': '../videos/source/3.mp4',
                          '4': '../videos/source/4.mp4',
                          '5': '../videos/source/5.mp4', }

        # Создание трекера на основе выбранного типа
        if self.types_map[tracker_type]:
            self.tracker = self.types_map[tracker_type]
        else:
            print("Error tracker_type. There is no such type. Types: 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', "
                  "'MOSSE', 'CSRT'")

        if self.video_map[video_choose]:
            self.video = cv2.VideoCapture(self.video_map[video_choose])
            self.ret, self.frame = self.video.read()

            if self.ret:
                self.frame_height, self.frame_width = self.frame.shape[:2]
                # print(self.frame_width, self.frame_height)
            else:
                print("The video was not read successfully")

            # Изменение размера видео для более удобного просмотра
            self.frame = cv2.resize(self.frame, [self.frame_width // 2, self.frame_height // 2])
        else:
            print("Error video_choose. There is no such video.")

        # Создание папки для сохранения видео (если ее нет)
        output_folder = f'../videos/{tracker_type}_data'
        os.makedirs(output_folder, exist_ok=True)

        # Инициализация записи видео для сохранения результатов
        output_file = os.path.join(output_folder, f'{tracker_type}_{video_choose}.avi')
        self.output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 60.0,
                                      (self.frame_width // 2, self.frame_height // 2), True)

    def render(self):
        if not self.ret:
            print('cannot read the video')
        # Select the bounding box in the first frame
        bbox = cv2.selectROI(self.frame, False)
        self.ret = self.tracker.init(self.frame, bbox)

        # Замер времени выполнения
        start_time = time.time()

        # Start tracking
        while True:
            ret, frame = self.video.read()
            if ret:
                frame = cv2.resize(frame, [self.frame_width // 2, self.frame_height // 2])
            if not ret:
                break
            timer = cv2.getTickCount()
            ret, bbox = self.tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            if ret:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(frame, self.tracker_type + " Tracker", (100, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.imshow("Tracking", frame)
            self.output.write(frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

        self.video.release()
        self.output.release()
        cv2.destroyAllWindows()

        # Завершение замера времени выполнения
        elapsed_time = time.time() - start_time
        print(f"Execution time ({self.tracker_type}): {elapsed_time:.4f} seconds")


if __name__ == '__main__':
    # Этот код будет выполнен только при вызове файла MultiTrackerCreator.py как скрипта
    key = 1
    if key == 0:
        TrackerCreator('KCF', '1').render()
        TrackerCreator('KCF', '2').render()
        TrackerCreator('KCF', '3').render()
        TrackerCreator('KCF', '4').render()
        TrackerCreator('KCF', '5').render()

    elif key == 1:
        TrackerCreator('CSRT', '1').render()
        TrackerCreator('CSRT', '2').render()
        TrackerCreator('CSRT', '3').render()
        TrackerCreator('CSRT', '4').render()
        TrackerCreator('CSRT', '5').render()


