from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:
    
    def __init__(self, data, popularity_rankings):
        self.rankings = popularity_rankings
        
        self.full_train_set = data.build_full_trainset()
        self.full_anti_test_set = self.full_train_set.build_anti_testset()
        
        self.train_set, self.test_set = train_test_split(data, test_size=0.25, random_state=1)
        
        # leave one out cross validation
        loocv = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in loocv.split(data):
            self.loocv_train = train
            self.loocv_test = test
            
        self.loocv_anti_test_set = self.loocv_train.build_anti_testset()
        
        sim_options = {'name': 'cosine', 'user_based': False}
        self.sims_algo = KNNBaseline(sim_options=sim_options)
        self.sims_algo.fit(self.full_train_set)
        
    def get_full_train_set(self):
        return self.full_train_set
    
    def get_full_anti_test_set(self):
        return self.full_anti_test_set
    
    def get_anti_test_set_for_user(self, test_subject):
        train_set = self.full_train_set
        fill = train_set.global_mean
        anti_test_set = []
        u = train_set.to_inner_uid(str(test_subject))
        user_items = set([j for (j, _) in train_set.ur[u]])
        anti_test_set += [(train_set.to_raw_uid(u), train_set.to_raw_iid(i), fill) for
                         i in train_set.all_items() if
                         i not in user_items]
        
        return anti_test_set
    
    def get_train_set(self):
        return self.train_set
    
    def get_test_set(self):
        return self.test_set
    
    def get_loocv_train_set(self):
        return self.loocv_train
        
    def get_loocv_test_set(self):
        return self.loocv_test
    
    def get_loocv_anti_test_set(self):
        return self.loocv_anti_test_set
    
    def get_similarities(self):
        return self.sims_algo
    
    def get_popularity_rankings(self):
        return self.rankings