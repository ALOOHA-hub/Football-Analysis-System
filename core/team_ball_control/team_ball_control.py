import numpy as np

class TeamBallControl:
    
    @staticmethod
    def calculate_team_ball_control(tracks):
        team_ball_control = []

        for frame_num, player_track in enumerate(tracks['players']):
            ball_idx = -1
            
            # Check if any player has the ball in this frame
            for player_id, track in player_track.items():
                if track.get('has_ball', False):
                    ball_idx = track['team']
                    break
            
            if ball_idx != -1:
                team_ball_control.append(ball_idx)
            else:
                # If no one has the ball, the last team to have it keeps control
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    # Suggestion: Add TEAM_CONTROL_NEUTRAL = 0 to constants
                    team_ball_control.append(0) 

        return np.array(team_ball_control)
