from moviepy.editor import VideoFileClip
import os

# Путь к папке с видео
video_folder = './videos/KCF_data'

# Папка для сохранения GIF
output_folder = './gifs'

# Перебор папок с видео
for root, dirs, files in os.walk(video_folder):
    for video_file in files:
        if video_file.endswith('.avi'):
            video_path = os.path.join(root, video_file)

            # Создание GIF с тем же именем видео
            gif_path = os.path.join(output_folder, os.path.splitext(video_file)[0] + '.gif')

            # Конвертация видео в GIF
            video_clip = VideoFileClip(video_path)
            video_clip.write_gif(gif_path, fps=60)  # Установите необходимые параметры FPS

            print(f'Видео {video_file} конвертировано в GIF: {gif_path}')
