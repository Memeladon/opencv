from methods.TrackerCreator import TrackerCreator

TrackerCreator('MEDIANFLOW', 'cars.mp4').render()
TrackerCreator('MEDIANFLOW', 'classic-red-spots-car.mp4').render()
TrackerCreator('MEDIANFLOW', 'front-of-cars-in-forest.mp4').render()
TrackerCreator('MEDIANFLOW', 'long-road-in-an-air').render()
TrackerCreator('MEDIANFLOW', 'two-cars-speeding').render()

# tracker = Tracker('MOSSE')
# tracker = Tracker('CSRT')