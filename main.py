from utils.video_utils import read_video, save_video, get_video_properties
from utils.config_loader import cfg
from utils.stub_manager import StubManager
import os
from core.detection import Detector
from core.trackers import Tracker
from core.annotation import Annotator
from core.team_assignment import TeamAssigner

def main():
    video_path = cfg['settings']['input_video_path']
    save_path = os.path.join(cfg['settings']['output_path'], 'output_video.mp4')
    model_path = cfg['settings']['model_path']
    stub_path = cfg['settings']['stub_path']

    frames = read_video(video_path)

    # Try loading cached tracks first (skips detection + tracking entirely)
    tracks = StubManager.load(stub_path)

    if tracks is None:
        # Step 1: Detect objects in frames
        detector = Detector(model_path)
        detections = detector.detect_frames(frames)

        # Step 2: Track objects across frames
        tracker = Tracker()
        tracks = tracker.get_object_tracks(detections)

        # Cache the tracks for next time
        StubManager.save(tracks, stub_path)

    # Step 3: Assign team colors (uses first frame to determine team clusters)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks["players"][0])

    # Step 4: Annotate frames and save video
    annotator = Annotator()
    frames = annotator.draw_annotations(frames, tracks)

    save_video(frames, save_path)

if __name__ == "__main__":
    main()