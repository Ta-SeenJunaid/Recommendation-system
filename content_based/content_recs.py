from movie_lens import MovieLens
from content_knn_algorithm import ContentKNNAlgorithm
from evaluator import Evaluator
from surprise import NormalPredictor

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.load_movie_lens_latest_small()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.get_popularity_ranks()
    return ml, data, rankings

np.random.seed(0)
random.seed(0)

(ml, evaluation_data, rankings) = LoadMovieLensData()

evaluator = Evaluator(evaluation_data, rankings)

random = NormalPredictor()
evaluator.add_algorithm(random, "Random")

content_KNN = ContentKNNAlgorithm()
evaluator.add_algorithm(content_KNN, "ContentKNN")

evaluator.evaluate(True)

evaluator.sample_top_n_recs(ml)