import cv2
import sys

video = cv2.VideoCapture(0)
fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
# Преобразуем числовой код формата в строку
fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
print(f"Формат читаемого видео: {fourcc_str}")


fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Выбираем кодек FFV1
out = cv2.VideoWriter('../../data_files/Миша2.avi', fourcc, 30, (int(video.get(3)), int(video.get(4))))

if not video.isOpened():
  print("Could not open video")
  sys.exit()

ok, frame = video.read()
if not ok:
  print('Cannot read video file')
  sys.exit()

while True:
    ok, frame = video.read()
    if not ok:
        break

    out.write(frame)


    cv2.imshow("Tracking", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27: break
