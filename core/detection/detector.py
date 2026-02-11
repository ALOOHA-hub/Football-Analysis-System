from ultralytics import YOLO

def Detector(self, model_path):
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, batch_size=20, conf=0.1):
        detections = []
        for i in range(0, len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=conf)
            detections += detections_batch
        return detections
        
        