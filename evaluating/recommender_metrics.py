
from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:
    
    def mae(predictions):
        return accuracy.mae(predictions, verbose=False)
    
    def rmse(predictions):
        return accuracy.rmse(predictions, verbose=False)
    
    def get_top_n(predictions, n=10, minimum_rating=4.0):
        top_n = defaultdict(list)
        
        for user_ID, movie_ID, actual_rating, estimated_rating, _ in predictions:
            if estimated_rating >= minimum_rating:
                top_n[int(user_ID)].append((int(movie_ID), estimated_rating))
                
        for user_ID, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[int(user_ID)] = ratings[:n]
        
        return top_n
    
    def hit_rate(top_n_predicted, left_out_predictions):
        hits = 0
        total = 0
        
        for left_out in left_out_predictions:
            user_ID = left_out[0]
            left_out_movie_ID = left_out[1]
            hit = False
            for movie_ID, predicted_rating in top_n_predicted[int(user_ID)]:
                if (int(left_out_movie_ID) == int(movie_ID)):
                    hit = True
                    break
            if hit:
                hits += 1
            
            total += 1
        
        return hits/total
    
    def cumulative_hit_rate(top_n_predicted, left_out_predictions, rating_cut_off):
        hits = 0
        total = 0
        
        for user_ID, left_out_movie_ID, actual_rating, \
        estimated_rating, _ in left_out_predictions:
            if actual_rating >= rating_cut_off:
                hit = False
                for movie_ID, predicted_rating in top_n_predicted[int(user_ID)]:
                    if int(left_out_movie_ID) == movie_ID:
                        hit = True
                        break
                if hit:
                    hits += 1
                total += 1
        
        return hits/total
    
    
    def rating_hit_rate(top_n_predicted, left_out_predictions):
        hits = defaultdict(float)
        total = defaultdict(float)
        
        for user_ID, left_out_movie_ID, actual_rating, \
            estimated_rating, _ in left_out_predictions:
                hit = False
                for movie_ID, predicted_rating in top_n_predicted[int(user_ID)]:
                    if int(left_out_movie_ID) == movie_ID:
                        hit = True
                        break
                if hit:
                    hits[actual_rating] += 1
                
                total[actual_rating] += 1
        
        for rating in sorted(hits.keys()):
            print(rating, hits[rating] / total[rating])
            
    
    def average_reciprocal_hit_rank(top_n_predicted, left_out_predictions):
        summation = 0
        total = 0
        
        for user_ID, left_out_movie_ID, actual_rating, \
            estimated_rating, _ in left_out_predictions:
                hit_rank = 0
                rank = 0
                for movie_ID, predicted_rating in top_n_predicted[int(user_ID)]:
                    rank = rank + 1
                    if int(left_out_movie_ID) == movie_ID:
                        hit_rank = rank 
                        break
                if hit_rank > 0:
                    summation += 1.0 / hit_rank
                
                total += 1
                
        return summation / total 
    
    