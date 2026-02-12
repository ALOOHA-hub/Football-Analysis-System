from sklearn.cluster import KMeans
from constants import (
    KMEANS_CLUSTERS,
    KMEANS_RANDOM_STATE,
    KMEANS_N_INIT_FIT,
    KMEANS_N_INIT_MODEL,
    TEAM_ID_1,
    TEAM_ID_2
)

class TeamAssigner:
    def __init__(self):
        self.team_colors = {} # store the color of each team
        self.player_team = {} # store the team of each player
        self.kmeans = None
    
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team:
            return self.player_team[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        predicted_label = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id = self.center_to_team_id_map[predicted_label]

        self.player_team[player_id] = team_id

        return team_id

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT_MODEL)
        kmeans.fit(image_2d)
        return kmeans

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_track in player_detections.items():
            bbox = player_track['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT_FIT)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        # Sort clusters by intensity (grayscale value) to ensure consistency (0=Darker, 1=Lighter)
        # Assuming Team 1 is the 'Lighter' team (e.g. White) and Team 2 is 'Darker' (e.g. Green)
        # Or vice versa. Sorting ensures determinism.
        
        centers = kmeans.cluster_centers_
        # Calculate intensity (mean of R,G,B)
        intensities = centers.mean(axis=1)
        
        # Sort indices: [index_of_darker, index_of_lighter]
        sorted_indices = intensities.argsort()

        # Assign sorted centers
        self.team_colors[TEAM_ID_1] = centers[sorted_indices[0]] # Darker/Team 1 (Green?)
        self.team_colors[TEAM_ID_2] = centers[sorted_indices[1]] # Lighter/Team 2 (White?)

        # We also need to map the KMEANS LABELS to these IDs.
        # But `predict` returns 0 or 1.
        # If we sorted the centers, we need to know which label (0 or 1) corresponds to which Sorted ID.
        # This requires overriding the predict logic or updating `player_team` map logic.
        # Actually easier: Just assign the colors to IDs here.
        # BUT `get_player_team` uses `kmeans.predict`.
        # `predict` returns the index of the closest center in `cluster_centers_`.
        # The `cluster_centers_` are NOT sorted by KMeans. They are as fitted.
        # So we just need to know: Is label 0 the darker one or lighter one?
        
        self.center_to_team_id_map = {}
        if sorted_indices[0] == 0: # Cluster 0 is darker
             # Map Cluster 0 -> Team 1 (Darker), Cluster 1 -> Team 2 (Lighter)
             self.center_to_team_id_map[0] = TEAM_ID_1
             self.center_to_team_id_map[1] = TEAM_ID_2
        else: # Cluster 1 is darker
             # Map Cluster 1 -> Team 1 (Darker), Cluster 0 -> Team 2 (Lighter)
             self.center_to_team_id_map[1] = TEAM_ID_1
             self.center_to_team_id_map[0] = TEAM_ID_2



        return self.team_colors

    def assign_team_to_tracks(self, video_frames, tracks):
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track_info in player_track.items():
                team = self.get_player_team(video_frames[frame_num], track_info['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = self.team_colors[team]

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        image = frame[int(y1):int(y2), int(x1):int(x2)]

        top_half_image = image[0:int(image.shape[0]/2), :]

        # Get the clusters and labels for each pixel
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_

        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)

        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color