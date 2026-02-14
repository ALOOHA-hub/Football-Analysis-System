# Football Analysis System ⚽🤖

A comprehensive Computer Vision pipeline for analyzing football matches. This system detects players, referees, and the ball; tracks them across frames; assigns team affiliations; calculates ball possession; and estimates player speed and distance covered.

## 🚀 How It Works

The system processes video input frame-by-frame through a modular pipeline:

1.  **Detection (`core/detection`)**:
    -   Uses **YOLOv8** to detect objects (Players, Referees, Ball, Goalkeepers).
    -   Trained on a custom football dataset.

2.  **Tracking (`core/trackers`)**:
    -   Uses **ByteTrack** algorithm to associate detections across frames.
    -   Assigns unique IDs to each player and maintains them throughout the video.

3.  **Team Assignment (`core/team_assignment`)**:
    -   Extracts player shirt colors using **KMeans Clustering**.
    -   Automatically distinguishes between two teams based on color intensity (Lighter vs Darker) to ensure consistent assignment.
    -   Goalkeepers are processed separately to avoid polluting team color data.

4.  **Ball Assignment (`core/player_ball_assignment`)**:
    -   Assigns the ball to the nearest player within a dynamic distance threshold.

5.  **Possession Control (`core/team_ball_control`)**:
    -   Calculates which team has possession at any given moment.
    -   Interpolates possession data to handle brief losses of control or tracking failures.

6.  **Speed & Distance (`core/speed_estimation`)**:
    -   Calculates player movement in pixels.
    -   Converts to real-world metrics (meters/sec, km/h) using a scaling factor (`constants/speed_estimator_consts.py`).

7.  **Visualization (`core/annotation`)**:
    -   Draws bounding boxes, ellipses, and text overlays.
    -   Displays live stats for speed, distance, and team possession percentages.

## 🧠 Key Challenge: Player Collision & Occlusion

One of the biggest challenges in multi-object tracking is handling **ID Switching** when players cross paths or occlude each other.

### Our Solution
We utilize **ByteTrack**, a state-of-the-art tracking algorithm that handles occlusion robustly:
1.  **Kalman Filter Prediction**: The tracker predicts where each player *should* be in the next frame based on their velocity.
2.  **Dual-Stage Matching**:
    -   **High Confidence**: Matches strong detections first.
    -   **Low Confidence**: Uses remaining (weaker) detections to recover "lost" tracks (e.g., partially occluded players).
3.  **Track Buffering**: If a player is completely hidden (e.g., behind another player), their track is kept alive in a "lost" buffer for up to **60 frames** (approx. 2 seconds). When they reappear near their predicted location, the ID is restored instead of starting a new one.

Additionally, we implement **Ball Interpolation** (`Tracker.interpolate_ball_positions`) to fill in missing ball detections when it is hidden by players.

## 📂 Project Structure

- `core/`: Contains the main logic modules (Detection, Tracking, Assignment, Annotation).
- `models/`: Stores the trained YOLOv8 model weights.
- `constants/`: Configuration files for colors, thresholds, and parameters.
- `utils/`: Helper functions for I/O and geometry calculations.
- `stubs/`: Cached tracking data to speed up development (avoids re-running YOLO every time).

## 🛠️ Installation & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Analysis**:
    ```bash
    python main.py
    ```
    -   The output video will be saved to `output_videos/`.
    -   Configuration settings can be adjusted in `config.yaml` or `constants/`.
