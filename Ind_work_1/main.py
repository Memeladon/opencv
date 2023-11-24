from Ind_work_1.legacy.PerfectCamShift import PerfectCamShift

# Запуск самодельного CamShift
print(__doc__)
import sys

try:
    video_src = sys.argv[1]
except:
    video_src = 0
PerfectCamShift(video_src).run()

