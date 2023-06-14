from movie_lens import MovieLens
from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from recommender_metrics import RecommenderMetrics

ml = MovieLens()

print("Loading movie ratings...")
data = ml.load_movie_lens_latest_small()

print("\nComputing movie popularity ranks so we can measure novelty later...")
rankings = ml.get_popularity_ranks()

print("\nComputing item similarities so we can measure diversity later...")
full_train_set = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based':False}
sims_algo = KNNBaseline(sim_options=sim_options)
sims_algo.fit(full_train_set)

print("\nBuilding recommendation model...")
train_set, test_set = train_test_split(data, test_size=0.25, random_state=1)

algo = SVD(random_state=10)
algo.fit(train_set)

print("\nComputing recommendations...")
predictions = algo.test(test_set)

print("\nEvaluating accuracy of model...")
print("RMSE: ", RecommenderMetrics.rmse(predictions))
print("MAE: ", RecommenderMetrics.mae(predictions))

print("\nEvaluating top-10 recommendations...")

# Set aside one rating per user for testing
LOOCV = LeaveOneOut(n_splits=1, random_state=1)

for train_set, test_set in LOOCV.split(data):
    print("Computing recommendations with leave-one-out...")
    
    # Train model without left-out ratings
    algo.fit(train_set)
    
    # Predicts ratings for left-out ratings only
    print("Predict ratings for left-out set...")
    left_out_predictions = algo.test(test_set)
    
    # Build predictions for all ratings not in the training set
    print("Predict all missing ratings...")
    big_test_set = train_set.build_anti_testset()
    all_predictions = algo.test(big_test_set)
    
    # Compute top 10 recs for each user
    print("Compute top 10 recs per user...")
    top_n_predicted = RecommenderMetrics.get_top_n(all_predictions, n=10)
    
    # See how often we recommended a movie the user actually rated
    print("\nHit Rate: ", RecommenderMetrics.hit_rate(top_n_predicted, left_out_predictions))
    
    # Break down hit rate by rating value
    print("\nrHR (Hit Rate by Rating value): ")
    RecommenderMetrics.rating_hit_rate(top_n_predicted, left_out_predictions)
    
    # See how often we recommended a movie the user actually liked
    print("\ncHR (Cumulative Hit Rate, rating >= 4): ", RecommenderMetrics.cumulative_hit_rate(top_n_predicted, left_out_predictions, 4.0))

    # Compute ARHR
    print("\nARHR (Average Reciprocal Hit Rank): ", RecommenderMetrics.average_reciprocal_hit_rank(top_n_predicted, left_out_predictions))
    
    
print("\nComputing complete recommendations, no hold outs...")
algo.fit(full_train_set)
big_test_set = full_train_set.build_anti_testset()
all_predictions = algo.test(big_test_set)
top_n_predicted = RecommenderMetrics.get_top_n(all_predictions, n=10)

# Print user coverage with a minimum predicted rating of 4.0:
print("\nUser coverage: ", RecommenderMetrics.user_coverage(top_n_predicted, full_train_set.n_users, rating_threshold=4.0))

# Measure diversity of recommendations:
print("\nDiversity: ",RecommenderMetrics().diversity(top_n_predicted, sims_algo))

# Measure novelty (average popularity rank of recommendations):
print("\nNovelty (average popularity rank): ", RecommenderMetrics.novelty(top_n_predicted, rankings))                                                  