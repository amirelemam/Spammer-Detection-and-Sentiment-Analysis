"""Spammer detection and sentiment analysis aiming to find people who need
need help in emergency situations
"""
from __future__ import division

import json
import logging
# import pdb
from copy import deepcopy

import matplotlib
# import matplotlib.pyplot as plt
from numpy import array
import psycopg2.extras
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB

logging.basicConfig(format='%(message)s', level=logging.DEBUG)

matplotlib.use('Agg')

class DataPreProcessing:
    """Get data from database and clean it"""

    def get_data_from_database_and_save_to_file(self, query, filename,
                                                isJson=True):
        # Connection between Python and PostgreSQL
        conn = psycopg2.connect(database='twitter', user='amir',
                                password='pg123@', host='143.107.137.90',
                                port='5432')
        # Create cursor
        cur = conn.cursor()
        # Execute the query
        cur.execute(query)
        # Create or empty output file
        with open(filename + ".txt", "w+"):
            pass
        # Save data to file
        if isJson:
            with open(filename + ".txt", "a") as _f:
                for row in cur.fetchall():
                    _f.write(json.dumps(row[0]) + '\n')
        else:
            with open(filename + ".txt", "a") as _f:
                for row in cur.fetchall():
                    _f.write(row[0] + "\n")

        # Close cursor and database connection
        cur.close()
        conn.close()
        return filename

    def get_manual_classified_tweets_from_database(self):
        """Get tweets manually classified from database
        and save them to a file
        """
        _query = """select p.json from paris_analise pa
                 inner join paris p on p.codtweet = pa.codtweet
                 """
        _filename = "trained_data"
        return self.get_data_from_database_and_save_to_file(_query, _filename)

    def get_manual_classification_from_database(self):
        """Get tweets' manual classification"""
        _query = """select pa.classificacao from paris_analise pa
                 inner join paris p on p.codtweet = pa.codtweet
                 """
        _filename = "classification"
        return self.get_data_from_database_and_save_to_file(_query, _filename,
                                                            isJson=False)

    def get_all_tweets_from_database(self):
        """Get all tweets from database"""
        _query = """SELECT json FROM paris"""
        _filename = "data"
        self.get_data_from_database_and_save_to_file(_query, _filename)

    def clean_data(self, input_filename, output_filename, isJson=True):
        """Transform data from file into a valid JSON"""
        # Create or empty output file
        with open(output_filename + ".txt", 'w+'):
            pass
        # Clean data and save it to output file
        with open(input_filename + ".txt", 'r') as _input, \
             open(output_filename + ".txt", 'a') as _output:
            if isJson:
                for line in _input:
                    line = line.replace('\\\\"', "@")
                    _output.write(str(line))
            else:
                for line in _input:
                    line = line.replace("\xc3\x83", "A")
                    _output.write(str(line))
        return output_filename

# class SampleCharacteristics:
#     """Displays and saves information about sample characteristics"""
#     def plot_and_save_graphics(self):
#         """"""

    # def display_general_characteristics(self):
    # """"""

    # def display_confusion_matrix(self):
    # """"""


class SpammerDetection:
    """Spammer detection based on user features"""

    def manual_classification(self, f_tweets, f_classifications):
        """Generate list with user ID and manual classification (SPAM, NAO
        SPAM, OUTRO)
        """

        manual_classification = []

        with open(f_tweets + ".txt", 'r') as tweets, \
             open(f_classifications + ".txt", 'r') as classifications:
            for raw_tweet, classification in zip(tweets, classifications):
                # Parse tweet from file to JSON format
                tweet = json.loads(raw_tweet)
                user_id = str(tweet['id'])
                manual_classification.append([user_id, classification.strip()])

        return manual_classification

    def get_criteria_features_by_user(self, f_tweets):
        """Generate list with user ID and boolean list of features that the
        user has/hasn't
        """

        all_users_features = []
        user_features = []

        with open(f_tweets + ".txt", 'r') as tweets:
            for raw_tweet in tweets:
                # Parse tweet from file to JSON format
                tweet = json.loads(raw_tweet)

                # Variables to nickname features
                following = int(tweet['user']['friends_count'])
                followers = int(tweet['user']['followers_count'])
                user_id = str(tweet['id'])
                geo = tweet['geo']
                favourites_count = int(tweet['user']['favourites_count'])
                num_hashtags = len(tweet['entities']['hashtags'])
                user_mentions = len(tweet['entities']['user_mentions'])
                source = tweet['source']
                total_tweets = int(tweet['user']['statuses_count'])
                user_favourited_by_someone = int(tweet['user']['listed_count'])

                # Fill list based on features

                # Number of followers > 30
                if following < 30:
                    user_features.append(1)
                else:
                    user_features.append(0)

                # Geolocation == true
                if geo is not None:
                    user_features.append(1)
                else:
                    user_features.append(0)

                # User included in another user's favorite
                if favourites_count == 0:
                    user_features.append(1)
                else:
                    user_features.append(0)

                # It has used a hashtag at least once
                if num_hashtags > 0:
                    user_features.append(1)
                else:
                    user_features.append(0)

                # Logged in on an iPhone
                if source is not None:
                    if "iPhone" not in source:
                        user_features.append(1)
                    else:
                        user_features.append(0)
                else:
                    user_features.append(0)

                # Mentioned by another user
                if user_mentions > 0:
                    user_features.append(1)
                else:
                    user_features.append(0)

                # User has less than 50 tweets
                if total_tweets > 50:
                    user_features.append(1)
                else:
                    user_features.append(0)

                # User has been included in another user's list
                if user_favourited_by_someone == 0:
                    user_features.append(1)
                else:
                    user_features.append(0)

                # Number of following is 2x or less Number of followers
                if followers*2 < following:
                    user_features.append(1)
                else:
                    user_features.append(0)

                all_users_features.append([user_id, user_features])

                user_features = []

        return all_users_features


    def get_prob_based_on_criteria(self, features):
        """Calculate the probability of user to be a spammer based on the
        criteria weights
        """

        criteria_weights = [0.53, 0.83, 0.85, 0.96, 0.917, 1, 0.01, 0.45, 0.5]
        users_prob_criteria = []

        for item in features:
            user_id = item[0]
            list_features = item[1]
            prob_spammer = 1

            # Multiply each feature by its correspondent weight
            prob_each_feature = array(list_features) * array(criteria_weights)

            # Multiply all probs to get the total prob
            for prob in prob_each_feature:
                if prob > 0:
                    prob_spammer *= prob

            users_prob_criteria.append([user_id, prob_spammer])

        return users_prob_criteria

    def remove_duplicates_prob_criteria(self, users_prob_criteria):
        """From list of users and its probabilities
        return a list without duplicated and the user probability set
        to the maximum probability in the list for that user
        """
        unique_prob = {}

        for user_id, prob in users_prob_criteria:
            if user_id in unique_prob:
                unique_prob[user_id] = max(unique_prob[user_id], prob)
            else:
                unique_prob[user_id] = prob

        unique_prob_criteria = []

        for user_id, prob in unique_prob.items():
            unique_prob_criteria.append([user_id, prob])

        return unique_prob_criteria

    def criteria_classification(self, users_prob_criteria):
        """Classifies user as spammer or not based on the paper criteria"""

        criteria_probable_spammers = set()

        for user_id, prob in users_prob_criteria:
            if prob > 0.009:
                criteria_probable_spammers.add(user_id)

        return list(criteria_probable_spammers)

    def trains_decision_tree_unweighted(self, features_list, classifications):
        """Trains Decision Tree Unweighted and returns classifier"""
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(array([features[1] for features in features_list]),
                      array([item[1] for item in classifications])
                     )
        return clf


    # def trains_decision_tree_weighted(self, features_list, classification):
    #     """Trains Decision Tree Weighted and returns classifier"""

    def trains_bernoulli_unweighted(self, features_list, classifications):
        """Trains Naive Bayes Bernoulli Unweighted and returns classifier"""
        clf = BernoulliNB()
        clf = clf.fit(array([features[1] for features in features_list]),
                      array([item[1] for item in classifications])
                     )
        return clf


    # def trains_bernoulli_weighted(self, features_list, classification):
    # """"""

    def classification(self, clf, data):
        """Classifies data received based on classifier given"""
        classification = []
        features = self.get_criteria_features_by_user(data)
        for feature in features:
            label = clf.predict(array(feature[1]))
            classification.append([deepcopy(feature[0]), deepcopy(label)])

        return classification

    def cross_validation_10_fold(self, clf, features_list, classifications):
        """Calculates mean of 10-fold cross validation"""
        data = array([features[1] for features in features_list])
        labels = array([item[1] for item in classifications])
        scores = cross_val_score(clf, data, labels, cv=10)

        return scores.mean()
# class SentimentAnalysis:
# """"""
#     def criteria_analysis(self):
#     """"""


class Main:
    """Initializes script"""
    def start(self):
        """Runs spammer detection and sentiment analysis"""
        data_pre_processing = DataPreProcessing()
        data_pre_processing.clean_data("paris300", "clean_300")

        spammer_detection = SpammerDetection()
        features = spammer_detection.get_criteria_features_by_user("clean_data")
        classification = spammer_detection.manual_classification("clean_data",
                                                                 "classification")
        clf = spammer_detection.trains_bernoulli_unweighted(features,
                classification)
        print(spammer_detection.cross_validation_10_fold(clf, features,
                                                         classification))
        # print(spammer_detection.classification(clf, "clean_300"))


if __name__ == '__main__':
    Main().start()

