from utils.video_utils import read_video, save_video, get_video_properties
from utils.config_loader import cfg
import os

def main():
    video_path = cfg['settings']['input_video_path']
    save_path = os.path.join(cfg['settings']['output_path'], 'output_video.mp4')
    if not os.path.exists(cfg['settings']['output_path']):
        os.makedirs(cfg['settings']['output_path'])
    frames = read_video(video_path)
    save_video(frames, save_path)
    fps, width, height = get_video_properties(video_path)
    print(f'FPS: {fps}, Width: {width}, Height: {height}')

if __name__ == "__main__":
    main()