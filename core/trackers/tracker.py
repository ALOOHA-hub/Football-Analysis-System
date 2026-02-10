from ultralytics import YOLO
from utils.config_loader import cfg
import supervision as sv
import cv2

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

        tracks = {
            'player': [], #list of dicts with track_id and bbox
            'ball': [], #list of dicts with track_id and bbox
            'referee': [], #list of dicts with track_id and bbox
        }

        #iterate over the frames
        for frame_number, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            #Convert detection to supervision fomrat
            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_id == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)       
            
            tracks['player'].append(detections_with_tracks[detections_with_tracks.class_id == cls_names_inv['player']])
            tracks['ball'].append(detections_with_tracks[detections_with_tracks.class_id == cls_names_inv['ball']])
            tracks['referee'].append(detections_with_tracks[detections_with_tracks.class_id == cls_names_inv['referee']])

            #store the tracks in the tracks dict
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist() 
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                #store the tracks in the tracks dict
                if cls_id == cls_names_inv['player']:
                    tracks['player'][frame_number][track_id] = [bbox]
                elif cls_id == cls_names_inv['referee']:
                    tracks['referee'][frame_number][track_id] = [bbox]

            #store the tracks in the tracks dict
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                #store the tracks in the tracks dict
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_number][track_id] = [bbox]
        
        return tracks
            
    def draw_annotations(self, frames, tracks):
        """
        Draw annotations on the frames
        """
        output_video_frames= []

        for frame_num, fram in enumerate(frames):
            frame = fram.copy()

            player_dict = tracks['player'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referee'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    def draw_ellipse(self, frame, bbox, color, track_id):
        """
        Draw an ellipse on the frame
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        radius_x = (x2 - x1) / 2
        radius_y = (y2 - y1) / 2
        cv2.ellipse(frame, (int(center_x), int(center_y)), (int(radius_x), int(radius_y)), 0, 0, 360, color, 2)
        cv2.putText(frame, str(track_id), (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame