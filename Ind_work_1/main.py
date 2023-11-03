from Ind_work_1.methods.PerfectCamShift import HandMadeCamShift

# Запуск самодельного CamShift
print(__doc__)
import sys

try:
    video_src = sys.argv[1]
except:
    video_src = 0
HandMadeCamShift(video_src).run()

