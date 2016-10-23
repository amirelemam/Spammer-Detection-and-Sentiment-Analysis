"""Spammer detection and sentiment analysis aiming to find people who need
help in emergency situations
"""
from __future__ import division

import json
# import pdb
import warnings
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import psycopg2.extras
from numpy import array
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix

matplotlib.use('Agg')

warnings.filterwarnings("ignore")

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

    def get_manual_classified_tweets_from_database(self, filename):
        """Get tweets manually classified from database
        and save them to a file
        """
        _query = """select p.json from paris_analise pa
                 inner join paris p on p.codtweet = pa.codtweet
                 """
        return self.get_data_from_database_and_save_to_file(_query, filename)

    def get_manual_classification_from_database(self, filename):
        """Get tweets' manual classification"""
        _query = """select pa.classificacao from paris_analise pa
                 inner join paris p on p.codtweet = pa.codtweet
                 """
        return self.get_data_from_database_and_save_to_file(_query, filename,
                                                            isJson=False)

    def get_all_tweets_from_database(self, filename):
        """Get all tweets from database"""
        _query = """SELECT json FROM paris"""
        # _filename = "data"
        self.get_data_from_database_and_save_to_file(_query, filename)

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

class SampleCharacteristics:
    """Displays and saves information about sample characteristics"""

    def plot_and_save_graphics(self, f_tweets):
        """"""
        count_tweets = 0
        total_unique_users = set()
        dates = []
        times = []

        with open(f_tweets + ".txt", "r") as f:
            for line in f:
                count_tweets += 1
                tweet = json.loads(line)
                total_unique_users.add(tweet['id'])

                item = str(tweet["created_at"])
                month = item[4:7]
                day = item[8:10]
                year = item[26:30]
                hour = item[11:13]
                minutes = item[14:16]
                seconds = item[17:19]
                dates.append(day + '/' + month + '/' + year)
                times.append(hour + ':' + minutes + ':' + seconds)

        tweets_per_user = list((map(int, total_unique_users).count(x)) for x in
                               map(int, total_unique_users))
        users_tweeted = list((tweets_per_user.count(x)) for x in
                             set(tweets_per_user))
        times_tweeted = list(set(tweets_per_user))
        # Count number of tweets/day
        dict_tweet_count = dict((x, dates.count(x)) for x in set(dates))
        list_tweet_count = list((dates.count(x)) for x in set(dates))

        # Plot bar graphic of tweets/user
        plt.bar(times_tweeted, users_tweeted, width=1, color="blue")
        plt.xlabel('# of tweets')
        plt.ylabel('Qty users tweeted')
        plt.title('# of tweets per user')
        plt.savefig("tweets_per_user.jpg")

        # Graphic of tweets/day
        # Plot bar graphic of tweets/day
        x = range(len(list_tweet_count))
        plt.bar(x, list_tweet_count, width=1, color="blue")
        plt.xlabel('Day')
        plt.ylabel('# of tweets')
        plt.title('# of tweets per day')
        plt.savefig("tweets_per_day.jpg")


    def display_general_characteristics(self, f_tweets):
        """"""
        count_tweets = 0
        total_unique_users = set()
        dates = []
        times = []

        with open(f_tweets + ".txt", "r") as f:
            for line in f:
                count_tweets += 1
                tweet = json.loads(line)
                total_unique_users.add(tweet['id'])

                item = str(tweet["created_at"])
                month = item[4:7]
                day = item[8:10]
                year = item[26:30]
                hour = item[11:13]
                minutes = item[14:16]
                seconds = item[17:19]
                dates.append(day + '/' + month + '/' + year)
                times.append(hour + ':' + minutes + ':' + seconds)

        # Count number of tweets
        print("Number of tweets: {}".format(count_tweets))

        # Count number of unique users
        print("Number of unique users: {}".format(len(set(total_unique_users))))

        # Average of tweets per user with graphic
        if len(set(total_unique_users)):
            avg = count_tweets / len(set(total_unique_users))
            print("Average of tweets per user: {:.1f}".format(avg))
        else:
            print("Average of tweets per user: 0")

        # Date/time range
        print("Date range: {} to {}".format(min(dates), max(dates)))
        print("Time range: {} to {}".format(min(times), max(times)))


class SpammerDetection:
    """Spammer detection based on user features"""

    criteria_weights = [0.53, 0.83, 0.85, 0.96, 0.917, 1, 0.01, 0.45, 0.5]

    def manual_classification(self, f_tweets, f_classifications):
        """Generate list with user ID and manual classification (SPAM, NAO
        SPAM)
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
        weights = self.criteria_weights
        users_prob_criteria = []

        for item in features:
            user_id = item[0]
            list_features = item[1]
            prob_spammer = 1

            # Multiply each feature by its correspondent weight
            prob_each_feature = array(list_features) * array(weights)

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

        for user_id, prob_total in users_prob_criteria:
            if 1 - prob_total < 0.009:
                criteria_probable_spammers.add(user_id)

        return list(criteria_probable_spammers)

    def trains_decision_tree(self, features_list, classifications):
        """Trains Decision Tree Unweighted and returns classifier"""
        clf = DecisionTreeClassifier()
        clf = clf.fit(array([features[1] for features in features_list]),
                      array([item[1] for item in classifications])
                     )
        return clf

    def trains_bernoulli(self, features_list, classifications):
        """Trains Naive Bayes Bernoulli Unweighted and returns classifier"""
        clf = BernoulliNB()
        clf = clf.fit(array([features[1] for features in features_list]),
                      array([item[1] for item in classifications])
                     )
        return clf

    def trains_multinomial(self, features_list, classifications):
        """Trains Naive Bayes Bernoulli Unweighted and returns classifier"""
        clf = MultinomialNB()
        clf = clf.fit(array([features[1] for features in features_list]),
                      array([item[1] for item in classifications])
                     )
        return clf

    def classification(self, clf, data):
        """Classifies data received based on classifier given"""
        classification = []
        features = self.get_criteria_features_by_user(data)
        for feature in features:
            label = clf.predict(array(feature[1]))
            classification.append([deepcopy(feature[0]), deepcopy(label[0])])

        return classification

    # def final_classification(self, bernoulli, decision_tree, multinomial, 
    #                          not_spam_data):
    #     for b, dt, m in zip(bernoulli, decision_tree, multinomial):


    def cross_validation_10_fold(self, clf, features_list, classifications):
        """Calculates mean of 10-fold cross validation"""
        data = array([features[1] for features in features_list])
        labels = array([item[1] for item in classifications])
        scores = cross_val_score(clf, data, labels, cv=10)

        return scores.mean()

    def confusion_matrix_(self, true, predicted):
        y_true = array([item[1] for item in true])
        y_pred = array([item[1] for item in predicted])
        labels = set(y_true)

        print("Row: truth")
        print("Column: predicted")
        print("\t{}".format(" ".join(labels)))
        for label, row in zip(labels, confusion_matrix(y_true, y_pred)):
            print("{}\t{}".format(label, row))


class SentimentAnalysis:
    """Does sentiment analysis on text"""

    sentiments = {'fear/anxiety': ['anxiety', 'anxious', 'catastrophic',
                                   'concern', 'disaster', 'emergency', 'fear',
                                   'insecure', 'panic', 'scared', 'terror',
                                   'threat', 'trouble', 'warning', 'worry'],
                  'shock': ['taken aback', 'aback', 'floor', 'god bless', 'omg',
                            'shock', 'stun', 'sudden', 'wtf', 'wth'],
                  'response': ['act', 'asap', 'escape', 'evacuate', 'flee',
                               'help', 'hide', 'run'],
                  'need information': ['breaking news', 'call', 'foul play',
                                       'incident', 'phone', 'report',
                                       'situation', 'unconfirmed'],
                  'threat': ['accident', 'attack', 'bomb', 'bullet', 'collapse',
                             'crash', 'explode', 'explosion', 'fire', 'gun',
                             'hijack', 'hit', 'hostage', 'plane', 'rifle',
                             'responsability', 'responsable', 'shoot', 'shot',
                             'struck', 'suicide', 'terrorism'],
                  'casualities': ['blood', 'body', 'bodies', 'corpses', 'dead',
                                  'corpse',  'injury', 'injure', 'kill',
                                  'wounded'],
                  'law enforcement': ['action', 'ambulance', 'command', 'medic',
                                      'operation', 'planes', 'police', 'cops',
                                      'FBI', 'security', 'recover', 'rescue',
                                      'response', 'restore', 'safe', 'safety',
                                      'save', 'shut', 'stay', 'survive',
                                      'suspend'],
        }

    def criteria_analysis(self, f_text, f_users_data):
        """Text sentiment analysis based on paper criteria"""

        sentiment_analysis = {"user_id": "",
                              "sentiments": set()}

        with open(f_text + ".txt", "r") as f:
            for line in f:
                tweet = json.loads(line)
                for key in self.sentiments.keys():
                    for keyword in range(len(self.sentiments[key])):
                        if self.sentiments[key][keyword] in tweet['text']:
                            sentiment_analysis['user_id'] = tweet['id']
                            sentiment_analysis['sentiments'].add(key)

        with open(f_users_data + ".txt", "w+") as f:
            f.write(str(sentiment_analysis))


class Main:
    """Initializes script"""
    def start(self):
        """Runs spammer detection and sentiment analysis"""

        # --- Filenames' Variables ---
        # Each variable defines a filename
        raw_data = "tweets"
        test_data = "paris300"
        clean_data = "clean_data"
        manual_classif_labels = "classification"
        manual_classif_raw_data = "trained_data"
        manual_classif_clean_data = "clean_cl_data"
        not_spam_data = "ns_tweets"
        sentiment_analysis = "sentiment_analysis"


        # --- Pre Processing Data ---
        dpp = DataPreProcessing()
        # Get tweets manually classified from database and save them to file
        dpp.get_manual_classified_tweets_from_database(manual_classif_raw_data)
        # Get classification given to tweets from database and save them to
        # file
        dpp.get_manual_classification_from_database(manual_classif_labels)
        # Get all tweets from database and save them to file
        dpp.get_all_tweets_from_database(raw_data)
        # Clean all tweets
        dpp.clean_data(raw_data, clean_data)
        # Clean tweets manually classified
        dpp.clean_data(manual_classif_raw_data, manual_classif_clean_data,
                       isJson=False)


        # --- Spam Detection ---
        sd = SpammerDetection()
        # Get all features from tweets
        features = sd.get_criteria_features_by_user(clean_data)
        # Makes list associating user id to manual classification
        classification = sd.manual_classification(manual_classif_clean_data,
                                                  manual_classif_labels)

        ## Training Classifiers

        # Trains classifier based on Decision Tree method
        clf_dt = sd.trains_decision_tree(features, classification)
        # Trains classifier based on Naive Bayes Multinomial method
        clf_m = sd.trains_multinomial(features, classification)
        # Trains classifier based on Naive Bayes Bernoulli method
        clf_b = sd.trains_bernoulli(features, classification)

        ## 10-Fold Cross Validation of classifiers

        # Do Cross Validation 10-Fold to determine accuracy of Decision Tree
        # Classifier
        accuracy_dt = sd.cross_validation_10_fold(clf_dt, features, 
                                                  classification)
        # Do Cross Validation 10-Fold to determine accuracy of Multinomial 
        # Classifier
        accuracy_m = sd.cross_validation_10_fold(clf_m, features,
                                                 classification)
        # Do Cross Validation 10-Fold to determine accuracy of Bernoulli 
        # Classifier
        accuracy_b = sd.cross_validation_10_fold(clf_b, features,
                                                 classification)

        print("10-Fold Cross Validation")
        print("Accuracy Bernoulli: {:.2f}".format(accuracy_b))
        print("Accuracy Multinomial: {:.2f}".format(accuracy_m))
        print("Accuracy Decision Tree: {:.2f}".format(accuracy_dt))

        ## Classifying data
        bernoulli = sd.classification(clf_b, clean_data)
        decision_tree = sd.classification(clf_dt, clean_data)
        multinomial = sd.classification(clf_m, clean_data)

        print("---- Confusion matrix ----")
        print("Bernoulli {}".format(sd.confusion_matrix_(classification,
                                                         bernoulli)))
        print("Multinomial {}".format(sd.confusion_matrix_(classification,
                                                           multinomial)))
        print("Decision Tree {}".format(sd.confusion_matrix_(classification,
                                                             decision_tree)))
        print("--------------------------\n")

        # Tweets are classified as "SPAM" or "NOT SPAM"
        # Tweets classified as "NOT SPAM" are saved to file
        # sd.final_classification(bernoulli, decision_tree, multinomial,
        #                         not_spam_data)

        # --- Sentiment Analysis ---
        s_analysis = SentimentAnalysis()
        # Analyze sentiments of non-spam tweets and save it + user ID to file
        s_analysis.criteria_analysis(not_spam_data, sentiment_analysis)


        # --- Sample Characteristics ---
        characteristics = SampleCharacteristics()
        # Saves general data characteristics to file
        characteristics.display_general_characteristics(clean_data)
        # Save data characteristics' graphics to file
        characteristics.plot_and_save_graphics(clean_data)

if __name__ == '__main__':
    Main().start()
