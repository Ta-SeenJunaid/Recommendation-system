from movie_lens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

test_subject = '70'
k = 10

ml = MovieLens()
data = ml.load_movie_lens_latest_small()

train_set = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': False}

model = KNNBasic(sim_options=sim_options)
model.fit(train_set)

sims_matrix = model.compute_similarities()

test_user_inner_ID = train_set.to_inner_uid(test_subject)

test_user_ratings = train_set.ur[test_user_inner_ID]
k_neighbors = heapq.nlargest(k, test_user_ratings, key = lambda t : t[1])

candidates = defaultdict(float)
for item_ID, rating in k_neighbors:
    similarity_row = sims_matrix[item_ID]
    for inner_ID, score in enumerate(similarity_row):
        candidates[inner_ID] += score * (rating / 5.0)
        
watched = {}
for item_ID, rating in train_set.ur[test_user_inner_ID]:
    watched[item_ID] = 1
    
pos = 0
for item_ID, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not item_ID in watched:
        movie_ID = train_set.to_raw_iid(item_ID)
        print(ml.get_movie_name(int(movie_ID)), rating_sum)
        pos += 1
        if pos > 10:
            break
