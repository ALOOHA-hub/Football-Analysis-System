import sys 
from utils.bbox_utils import get_center_of_bbox, measure_distance
from constants import MAX_PLAYER_BALL_DISTANCE

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = MAX_PLAYER_BALL_DISTANCE
    
    def assign_ball_to_players(self, tracks):
        for frame_num, player_track in enumerate(tracks['players']):
            ball_track = tracks['ball'][frame_num]
            if 1 not in ball_track:
                continue

            ball_bbox = ball_track[1]['bbox']
            assigned_player = self.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                tracks['ball'][frame_num][1]['assigned_to'] = assigned_player

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        return assigned_player