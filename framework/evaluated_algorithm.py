from recommender_metrics import RecommenderMetrics
# from evaluation_data import EvaluationData

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def evaluate(self, evaluation_data, do_top_n, n = 10, verbose = True):
        metrics = {}
        if verbose:
            print("Evaluating accuracy....")
        self.algorithm.fit(evaluation_data.get_train_set())
        predictions = self.algorithm.test(evaluation_data.get_test_set())
        metrics["RMSE"] = RecommenderMetrics.rmse(predictions)
        metrics["MAE"] = RecommenderMetrics.mae(predictions)
        
        if do_top_n:
            if verbose:
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluation_data.get_loocv_train_set())
            left_out_predictions = self.algorithm.test(evaluation_data.get_loocv_test_set())
            all_predictions = self.algorithm.test(evaluation_data.get_loocv_anti_test_set())
            top_n_predicted = RecommenderMetrics.get_top_n(all_predictions, n)
            if verbose:
                print("Computing hit-rate and rank metrics...")
            metrics["HR"] = RecommenderMetrics.hit_rate(top_n_predicted, left_out_predictions)
            metrics["CHR"] = RecommenderMetrics.cumulative_hit_rate(top_n_predicted, left_out_predictions)
            metrics["ARHR"] = RecommenderMetrics.average_reciprocal_hit_rank(top_n_predicted, left_out_predictions)
            
            if verbose:
                print("Computing recommendations with full data set ...")
            self.algorithm.fit(evaluation_data.get_full_train_set())
            all_predictions = self.algorithm.test(evaluation_data.get_full_anti_test_set())
            top_n_predicted = RecommenderMetrics.get_top_n(all_predictions, n)
            if verbose:
                print("Analyzing coverage, diversity and novelty....")
            metrics["Coverage"] = RecommenderMetrics.user_coverage(top_n_predicted, evaluation_data.get_full_train_set().n_users,
                                                                   rating_threshold=4.0)
            metrics["Diversity"] = RecommenderMetrics.diversity(top_n_predicted, evaluation_data.get_similarities())
            metrics["Novelty"] = RecommenderMetrics.novelty(top_n_predicted, 
                                                            evaluation_data.get_popularity_rankings())
            
        if verbose:
            print("Analysis complete.....")
        
        return metrics
    
    def get_name(self):
        return self.name
    
    def get_algorithm(self):
        return self.algorithm