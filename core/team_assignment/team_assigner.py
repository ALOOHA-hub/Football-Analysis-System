from sklearn.cluster import KMeans



class TeamAssigner:
    def __init__(self):
        self.team_colors = {} # store the color of each team
        self.player_team = {} # store the team of each player
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team:
            return self.player_team[player_id]
        else:
            player_color = self.get_player_color(frame, player_bbox)
            self.player_team[player_id] = player_color

            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1 #+1 just to make the team id 1 and 2 instead of 0 and 1 
            self.player_team[player_id] = team_id

            return team_id

    def get_culstering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(image_2d)

        return kmeans

    def assign_team_color(self, frame, player_detections):
        
        player_colors = []
        for _, player_detections in player_detections.items():
            bbox = player_detections['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, random_state=0, n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

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

        #Get the clusters and labels for each pixel
        kmeans = self.get_culstering_model(top_half_image)
        labels = kmeans.labels_

        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)

        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

        

        

    