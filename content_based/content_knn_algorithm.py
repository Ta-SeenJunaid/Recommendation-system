from surprise import AlgoBase
from surprise import PredictionImpossible
from movie_lens import MovieLens
import math
import numpy as np
import heapq


class ContentKNNAlgorithm(AlgoBase):
    
    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        ml = MovieLens()
        genres = ml.get_genres()
        years = ml.get_years()
        # mes = ml.get_mise_en_scene()
        
        print("Computing content-based similarity matrix...")
        
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for this_rating in range(self.trainset.n_items):
            if this_rating % 100 == 0:
                print(this_rating, " of", self.trainset.n_items)
            for other_rating in range(this_rating + 1, self.trainset.n_items):
                this_movie_ID = int(self.trainset.to_raw_iid(this_rating))
                other_movie_ID = int(self.trainset.to_raw_iid(other_rating))
                genre_similarity = self.compute_genre_similarity(this_movie_ID, other_movie_ID, genres)
                year_similarity = self.compute_year_similarity(this_movie_ID, other_movie_ID, years)
                # mes_similarity = self.compute_mise_en_scene_similarity(this_movie_ID, other_movie_ID, mes)
                self.similarities[this_rating, other_rating] = genre_similarity * year_similarity
                self.similarities[other_rating, this_rating] = self.similarities[this_rating, other_rating]
        
        print("...done...")
        return self
    
    def compute_genre_similarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
            
        return sumxy/math.sqrt(sumxx*sumyy)
    
    
    def compute_year_similarity(self, movie1, movie2, years):
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff / 10.0)
        return sim
    
    def compute_mise_en_scene_similarity(self, movie1, movie2, mes):
        mes1 = mes[movie1]
        mes2 = mes[movie2]
        if mes1 and mes2:
            shot_length_diff = math.fabs(mes1[0] - mes2[0])
            color_variance_diff = math.fabs(mes1[1] - mes2[1])
            motion_diff = math.fabs(mes1[3] - mes2[3])
            lighting_diff = math.fabs(mes1[5] - mes2[5])
            num_shots_diff = math.fabs(mes1[6] - mes2[6])
            return shot_length_diff * color_variance_diff * motion_diff * lighting_diff * num_shots_diff
        else:
            return 0
        
    
    def estimate(self, u, i):
        
        if not self.trainset.knows_user(u) and self.trainset.knows_item(i):
            raise PredictionImpossible("User and/or item is unknown.")
            
        neighbors = []
        for rating in self.trainset.ur[u]:
            genre_similarity = self.similarities[i, rating[0]]
            neighbors.append((genre_similarity, rating[1]))
            
        k_neighbors = heapq.nlargest(self.k, neighbors, key = lambda t: t[0])
        
        sim_total = weighted_sum = 0
        for (sim_score, rating) in k_neighbors:
            if sim_score > 0:
                sim_total += sim_score
                weighted_sum += sim_score * rating
                
        if sim_total == 0:
            raise PredictionImpossible("No neighbors")
            
        predicted_rating = weighted_sum / sim_total
        
        return predicted_rating
