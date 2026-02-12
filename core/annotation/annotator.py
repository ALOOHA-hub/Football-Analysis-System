import cv2
import numpy as np
import supervision as sv
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from constants import (
    PLAYER_DEFAULT_COLOR,
    BALL_POSSESSION_COLOR,
    REFEREE_COLOR,
    BALL_COLOR,
    TEXT_COLOR,
    LABEL_RECTANGLE_WIDTH,
    LABEL_RECTANGLE_HEIGHT,
    LABEL_Y_OFFSET,
    LABEL_X_TEXT_OFFSET,
    LABEL_TEXT_SHIFT_THRESHOLD,
    ELLIPSE_HEIGHT_RATIO,
    ELLIPSE_START_ANGLE,
    ELLIPSE_END_ANGLE,
    ELLIPSE_THICKNESS,
    TRIANGLE_SIZE,
    TRIANGLE_Y_OFFSET,
    FONT_SCALE,
    FONT_THICKNESS,
    TEAM_ID_1,
    TEAM_ID_2,
    OVERLAY_RECT_START,
    OVERLAY_RECT_END,
    OVERLAY_COLOR,
    OVERLAY_ALPHA,
    OVERLAY_TEXT_POS_1,
    OVERLAY_TEXT_POS_2,
    OVERLAY_TEXT_COLOR,
    OVERLAY_FONT_SCALE,
    OVERLAY_FONT_THICKNESS
)

class Annotator:

    def __init__(self):
        pass
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):

        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(ELLIPSE_HEIGHT_RATIO * width)),
            angle=0.0,
            startAngle=ELLIPSE_START_ANGLE,
            endAngle=ELLIPSE_END_ANGLE,
            color=color,
            thickness=ELLIPSE_THICKNESS,
            lineType=cv2.LINE_4
        )
        
        x1_rect = x_center - LABEL_RECTANGLE_WIDTH // 2
        x2_rect = x_center + LABEL_RECTANGLE_WIDTH // 2
        y1_rect = (y2 - LABEL_RECTANGLE_HEIGHT // 2) + LABEL_Y_OFFSET
        y2_rect = (y2 + LABEL_RECTANGLE_HEIGHT // 2) + LABEL_Y_OFFSET
        
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + LABEL_X_TEXT_OFFSET
            if track_id > LABEL_TEXT_SHIFT_THRESHOLD:
                x1_text -= 10
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y],
            [x - TRIANGLE_SIZE, y - TRIANGLE_Y_OFFSET],
            [x + TRIANGLE_SIZE, y - TRIANGLE_Y_OFFSET],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, TEXT_COLOR, 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, OVERLAY_RECT_START, OVERLAY_RECT_END, OVERLAY_COLOR, cv2.FILLED)
        alpha = OVERLAY_ALPHA
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
        # Get the number of times each team had control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==TEAM_ID_1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==TEAM_ID_2].shape[0]
        
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Control: {team_1*100:.2f}%", OVERLAY_TEXT_POS_1, cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OVERLAY_TEXT_COLOR, OVERLAY_FONT_THICKNESS)
        cv2.putText(frame, f"Team 2 Control: {team_2*100:.2f}%", OVERLAY_TEXT_POS_2, cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OVERLAY_TEXT_COLOR, OVERLAY_FONT_THICKNESS)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", PLAYER_DEFAULT_COLOR)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], BALL_POSSESSION_COLOR)
            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], REFEREE_COLOR)
            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], BALL_COLOR)

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)
        return output_video_frames