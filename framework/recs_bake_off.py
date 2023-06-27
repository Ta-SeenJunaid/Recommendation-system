from movie_lens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings....")
    data = ml.load_movie_lens_latest_small()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.get_popularity_ranks()
    return (data, rankings)


np.random.seed(0)
random.seed(0)

(evaluation_data, rankings) = LoadMovieLensData()

evaluator = Evaluator(evaluation_data, rankings)

svd_algorithm = SVD(random_state=10)
evaluator.add_algorithm(svd_algorithm, "SVD")

random = NormalPredictor()
evaluator.add_algorithm(random, "Random")

evaluator.evaluate(True)

