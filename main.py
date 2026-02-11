from utils.video_utils import read_video, save_video, get_video_properties
from utils.config_loader import cfg
import os
from core.trackers import Tracker

def main():
    video_path = cfg['settings']['input_video_path']
    save_path = os.path.join(cfg['settings']['output_path'], 'output_video.mp4')
    model_path = cfg['settings']['model_path']
    stub_path = cfg['settings']['stub_path']

    frames = read_video(video_path)

    #Init Tracker
    tracker = Tracker(model_path)
        
    tracks = tracker.get_object_tracks(frames,
                                       read_from_stub=True,
                                       stub_path=stub_path)
    # Save video with tracks
    frames = tracker.draw_annotations(frames, tracks)
    save_video(frames, save_path)

if __name__ == "__main__":
    main()