import supervision as sv
import pickle
import os
import numpy as np
import cv2
import sys 
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position
from ultralytics import YOLO
from constants import (
    CLASS_PLAYER, 
    CLASS_REFEREE, 
    CLASS_BALL, 
    CLASS_GOALKEEPER,
    TRACKER_ACTIVATION_THRESHOLD,
    TRACKER_LOST_BUFFER
)
import pandas as pd

class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=TRACKER_ACTIVATION_THRESHOLD, 
            lost_track_buffer=TRACKER_LOST_BUFFER
        )

    def get_object_tracks(self, detections):
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            #Convert from the ultralytics format to the supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert goalKeeper to player because each goalkeaper is player and there is not enough data to train a model for goalKeeper
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == CLASS_GOALKEEPER:
                    detection_supervision.class_id[object_ind] = cls_names_inv[CLASS_PLAYER]
            
            #Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv[CLASS_PLAYER]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv[CLASS_REFEREE]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv[CLASS_BALL]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Interpolate Ball Positions
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])

        return tracks

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
        
    @staticmethod
    def add_position_to_tracks(tracks):
        from utils.bbox_utils import get_center_of_bbox, get_foot_position
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == CLASS_BALL:
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position