import sys
from utils.bbox_utils import measure_distance
from constants import SPEED_FRAME_WINDOW, SPEED_FRAME_RATE, METERS_PER_PIXEL

class SpeedEstimator:
    def __init__(self):
        self.frame_window = SPEED_FRAME_WINDOW
        self.frame_rate = SPEED_FRAME_RATE
        self.meters_per_pixel = METERS_PER_PIXEL

    def speed_and_distance_to_tracks(self, tracks):
        """
        Calculate speed and distance for each player in the tracks.
        
        Args:
            tracks (dict): Dictionary of tracks.
        """
        
        total_distance = {}

        for obj, obj_tracks in tracks.items():
            if obj == "ball" or obj == "referees":
                continue
            
            number_of_frames = len(obj_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames-1)

                for track_id, _ in obj_tracks[frame_num].items():
                    if track_id not in obj_tracks[last_frame]:
                        continue

                    start_position = obj_tracks[frame_num][track_id]['position']
                    end_position = obj_tracks[last_frame][track_id]['position']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_pixels = measure_distance(start_position, end_position)
                    distance_meters = distance_pixels * self.meters_per_pixel
                    
                    speed_meters_per_second = distance_meters / (self.frame_window / self.frame_rate)
                    speed_kmh = speed_meters_per_second * 3.6

                    if obj not in total_distance:
                        total_distance[obj] = {}
                    
                    if track_id not in total_distance[obj]:
                        total_distance[obj][track_id] = 0
                    
                    total_distance[obj][track_id] += distance_meters

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id in obj_tracks[frame_num_batch]:
                            obj_tracks[frame_num_batch][track_id]['speed'] = speed_kmh
                            obj_tracks[frame_num_batch][track_id]['distance'] = total_distance[obj][track_id]

                    
                    
                    
                
            
            
            