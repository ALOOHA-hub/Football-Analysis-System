from ultralytics import YOLO
from utils.config_loader import cfg
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        """
        Initialize the tracker
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def track(self, frames):
        """
        Track objects in the frames
        """
        batch_size = 20
        detections = []
        
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i: i+batch_size], conf=cfg['settings']['confidence_threshold'], iou=cfg['settings']['iou_threshold']) #here we could use track insted
            detections+=detections_batch
            #just for testing issues
            break
        return detections

    def get_object_tracks(self, frames):
        """
        Get object tracks from the detections
        """
        detections = self.track(frames)

        for frame_number, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            #Convert detection to supervision fomrat
            detection_supervision = sv.Detections.from_ultralytics(detection)

            print(detection_supervision)
            

        