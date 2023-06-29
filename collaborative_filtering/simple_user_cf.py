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
               'user_based': True}

model = KNNBasic(sim_options=sim_options)
model.fit(train_set)

sims_matrix = model.compute_similarities()

test_user_inner_ID = train_set.to_inner_uid(test_subject)
similarity_row = sims_matrix[test_user_inner_ID]

similar_users = []
for inner_ID, score in enumerate(similarity_row):
    if inner_ID != test_user_inner_ID:
        similar_users.append( (inner_ID, score))
        
k_neighbors = heapq.nlargest(k, similar_users, key=lambda t: t[1])

candidates = defaultdict(float)
for similar_user in k_neighbors:
    inner_ID = similar_user[0]
    user_similarity_score = similar_user[1]
    their_ratings = train_set.ur[inner_ID]
    for rating in their_ratings:
        candidates[rating[0]] += (rating[1] / 5.0) * user_similarity_score
        
watched ={}
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
