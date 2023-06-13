import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np

class MovieLens:
    
    movieID_to_name = {}
    name_to_movieID = {}
    ratings_path = '../ml-latest-small/ratings.csv'
    movies_path = '../ml-latest-small/movies.csv'
    
    def load_movie_lens_latest_small(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        
        ratings_dataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}
        
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        
        ratings_dataset = Dataset.load_from_file(self.ratings_path, reader=reader)
        
        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)
            for row in movie_reader:
                movie_ID = int(row[0])
                movie_name = row[1]
                self.movieID_to_name[movie_ID] = movie_name
                self.name_to_movieID[movie_name] = movie_ID
        
        return ratings_dataset
    
    
    def get_user_ratings(self, user):
        user_ratings = []
        hit_user = False
        with open(self.ratings_path, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)
            for row in rating_reader:
                userID = int(row[0])
                if user == userID:
                    movie_ID = int(row[1])
                    rating = float(row[2])
                    user_ratings.append((movie_ID, rating))
                    hit_user = True
                if hit_user and user != userID:
                    break
            
            return user_ratings
        
        
    def get_popularity_ranks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratings_path, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)
            for row in rating_reader:
                movie_ID = int(row[1])
                ratings[movie_ID] += 1
        rank = 1
        for movie_ID, rating_count in \
            sorted(ratings.items(), key=lambda x: x[1], reverse=True):
                rankings[movie_ID] = rank
                rank += 1
        return rankings
        
        
    def get_genres(self):
        genres = defaultdict(list)
        genreIDs = {}
        max_genreID = 0
        with open(self.movie_path, newline='', encoding='ISO-8859-1') as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)
            for row in movie_reader:
                movieID = int(row[0])
                genre_list = row[2].split('|')
                genreID_list = []
                for genre in genre_list:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = max_genreID
                        genreIDs[genre] = genreID
                        max_genreID += 1
                    genreID_list.append(genreID)
                genres[movieID] = genreID_list
        
        for (movieID, genreID_list) in genres.items():
            bitfield = [0] * max_genreID
            for genreID in genreID_list:
                bitfield[genreID] = 1
            genres[movieID] = bitfield
        
        return genres
    
    def get_years(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)
            for row in movieReader:
                movieID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[movieID] = int(year)
        return years
    
    def get_mise_en_scene(self):
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mes_reader = csv.reader(csvfile)
            next(mes_reader)
            for row in mes_reader:
                movieID = int(row[0])
                avg_shot_length = float(row[1])
                mean_color_variance = float(row[2])
                stddev_color_variance = float(row[3])
                mean_motion = float(row[4])
                stddev_motion = float(row[5])
                mean_lighting_key = float(row[6])
                num_shots = float(row[7])
                mes[movieID] = [avg_shot_length, mean_color_variance, stddev_color_variance,
                   mean_motion, stddev_motion, mean_lighting_key, num_shots]
        return mes
    
    def get_movie_name(self, movieID):
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""
        
    def get_movieID(self, movie_name):
        if movie_name in self.name_to_movieID:
            return self.name_to_movieID[movie_name]
        else:
            return 0