"""Spammer detection and sentiment analysis aiming to find people who need
help in emergency situations
"""
from __future__ import division

# from builtins import zip
# from builtins import map
# from builtins import str
# from builtins import range
# from builtins import object

from datetime import datetime
import json
# import pdb
import sys
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

_time = datetime.now()
log_file = 'log_' + str(_time.day) + '-' + str(_time.month) + '-' + str(_time.year) + '_' + str(_time.hour) + ' ' + str(_time.minute) + ' ' + str(_time.second) + '.txt'


# tweets_problem = 'tweets_problem' + sys.argv[1]
with open(log_file, 'w+') as log:
    pass

class DataPreProcessing(object):
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
        """
        Get tweets manually classified from database and save them to a file
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
                for row in _input:
                    row = row.replace('\\\\"', "@")
                    _output.write(str(row))
            else:
                for row in _input:
                    row = row.replace("\xc3\x83", "A")
                    _output.write(str(row))
        return output_filename


class SampleCharacteristics(object):
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
        list_tweet_count = list((dates.count(x)) for x in set(dates))

        # Plot bar graphic of tweets/user
        plt.bar(times_tweeted, users_tweeted, width=1, color="blue")
        plt.xlabel('# of tweets')
        plt.ylabel('Qty users tweeted')
        plt.title('# of tweets per user')
        plt.savefig("tweets_per_user_" + sys.argv[1] + ".jpg")

        # Graphic of tweets/day
        # Plot bar graphic of tweets/day
        x = list(range(len(list_tweet_count)))
        plt.bar(x, list_tweet_count, width=1, color="blue")
        plt.xlabel('Day')
        plt.ylabel('# of tweets')
        plt.title('# of tweets per day')
        plt.savefig("tweets_per_day_" + sys.argv[1] + ".jpg")

    def display_general_characteristics(self, f_tweets):
        """"""
        count_tweets = 0
        total_unique_users = set()
        dates = []
        times = []

        with open(f_tweets + ".txt", "r") as f:
            for line in f:
                try:
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
                except:
                    pass
        # Count number of tweets
        print("Number of tweets: {}".format(count_tweets))
        with open(log_file, 'a') as log:
            log.write("Number of tweets: " + str(count_tweets) + '\n')

        # Count number of unique users
        print("Number of unique users: {}".format(len(set(total_unique_users))))
        with open(log_file, 'a') as log:
            log.write("Number of unique users: " + str(len(set(total_unique_users))) + '\n')

        # Average of tweets per user with graphic
        if len(set(total_unique_users)):
            avg = count_tweets / len(set(total_unique_users))
            print("Average of tweets per user: {:.1f}".format(avg))
            with open(log_file, 'a') as log:
                log.write("Average of tweets per user: " + str.format("{:.1f}", avg) + '\n')
        else:
            print("Average of tweets per user: 0")
            with open(log_file, 'a') as log:
                log.write("Average of tweets per user: 0\n")

        # Date/time range
        with open(log_file, 'a') as log:
            print("\nDate range: {} to {}".format(min(dates), max(dates)))
            log.write("Date range: " + str(min(dates)) + " to " + str(max(dates)))
            print("Time range: {} to {}".format(min(times), max(times)))
            log.write("\nTime range: " + str(min(times)) + " to " + str(max(times)))

    def classification_results(self, classification, title):
        spam = 0
        not_spam = 0
        total_tweets = 0

        for item in classification:
            if item[1] == "SPAM":
                spam += 1
            else:
                not_spam += 1
            total_tweets += 1

        with open(log_file, 'a') as log:
            print("\n{}".format(title))
            log.write("\n" + title)
            print("Spam: {} ({:.1f}%)".format(spam, spam/total_tweets*100))
            log.write("\nSpam: " + str(spam) + " (" + str.format("{:.1f}", spam/total_tweets*100) + "%)")
            print("Not Spam: {} ({:.1f}%)".format(not_spam, not_spam/total_tweets*100))
            log.write("\nNot Spam: " + str(not_spam) + " (" + str.format("{:.1f}%)", not_spam/total_tweets*100) + ")%")


class SpammerDetection(object):
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
        count = 0

        with open(f_tweets + ".txt", 'r') as tweets: #, open(tweets_problem + '.txt', 'a') as p:
            for raw_tweet in tweets:
                try:
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
                except:
                    pass
                    # p.write(raw_tweet)

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

        for user_id, prob in list(unique_prob.items()):
            unique_prob_criteria.append([user_id, prob])

        return unique_prob_criteria

    def criteria_classification(self, users_prob_criteria):
        """Classifies user as spammer or not based on the paper criteria"""

        classification = []

        for user_id, prob_total in users_prob_criteria:
            if 1 - prob_total < 0.991:
                classification.append([user_id, "SPAM"])
            else:
                classification.append([user_id, "NAO SPAM"])

        return classification

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

        user_id__features = self.get_criteria_features_by_user(data)
        for line in user_id__features:
            label = clf.predict(array(line[1]))
            classification.append([deepcopy(line[0]), deepcopy(label[0])])

        return classification

    def final_classification(self, criteria, bernoulli, decision_tree,
                                 multinomial):
        final_classification = []
        for c, b, dt, m in zip(criteria, bernoulli, decision_tree, multinomial):
            classification = []
            classification.append(c[1])
            classification.append(b[1])
            classification.append(dt[1])
            classification.append(m[1])

            if classification.count('NAO SPAM') > 2:
                final_classification.append([deepcopy(c[0]), 'NAO SPAM'])
            elif classification.count('NAO SPAM') == 2:
                final_classification.append([deepcopy(c[0]), deepcopy(c[1])])
            else:
                final_classification.append([deepcopy(c[0]), 'SPAM'])

        return final_classification

    def cross_validation_10_fold(self, clf, features_list, classifications):
        """Calculates mean of 10-fold cross validation"""
        data = array([features[1] for features in features_list])
        labels = array([item[1] for item in classifications])
        scores = cross_val_score(clf, data, labels, cv=10)

        return scores.mean()

    def confusion_matrix_(self, true, predicted, title):
        y_true = array([item[1] for item in true])
        y_pred = array([item[1] for item in predicted])
        labels = set(y_true)

        with open(log_file, 'a') as log:
            print("Method: {}".format(title))
            log.write("\nMethod: " + title)
            print("Row: truth")
            log.write("\nRow: truth")
            print("Column: predicted")
            log.write("\nColumn: predicted")
            print("\t{}".format(" ".join(labels)))
            log.write("\n\t" + (" ".join(labels)))
            for label, row in zip(labels, confusion_matrix(y_true, y_pred)):
                print("{}\t{}".format(label, row))
                log.write('\n' + str(label) + "\t" + str(row))

    def criteria_accuracy(self, predicted, true):
        spam_pred = [item[1] for item in predicted].count('SPAM')
        not_spam_pred = [item[1] for item in predicted].count("NAO SPAM")
        spam_true = [item[1] for item in true].count('SPAM')
        not_spam_true = [item[1] for item in true].count("NAO SPAM")

        if spam_true > 0:
            accuracy_spam = 1 - (abs(spam_pred - spam_true) / spam_true)
        else:
            accuracy_spam = 0

        if not_spam_true > 0:
            accuracy_not_spam = 1 - (abs(not_spam_pred - not_spam_true) / not_spam_true)
        else:
            accuracy_not_spam = 0

        accuracy = (accuracy_spam + accuracy_not_spam) / 2
        print("Accuracy Criteria: {:.2f}".format(accuracy))
        print("\tCriteria True Spam: {}".format(spam_true))
        print("\tCriteria Predicted Spam: {}".format(spam_pred))
        print("\tCriteria True Not Spam: {}".format(not_spam_true))
        print("\tCriteria Predicted Not Spam: {}".format(not_spam_pred))
        with open(log_file, 'a') as log:
            log.write("\nAccuracy Criteria: {:.2f}".format(accuracy))
            log.write("\n\tCriteria True Spam: " + str(spam_true))
            log.write("\n\tCriteria Predicted Spam: " + str(spam_pred))
            log.write("\n\tCriteria True Not Spam: " + str(not_spam_true))
            log.write("\n\tCriteria Predicted Not Spam: " + str(not_spam_pred))

class SentimentAnalysis(object):
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

    def sentiment_analysis(self, user_id, tweet, f_output):
        sentiment_analysis = set()

        for key in list(self.sentiments.keys()):
            for keyword in range(len(self.sentiments[key])):
                if self.sentiments[key][keyword] in tweet:
                    sentiment_analysis.add(key)
            if len(list(sentiment_analysis)) > 0:
                with open(f_output + ".txt", "a") as f:
                    f.write(str(user_id))
                    f.write('\t')
                    f.write(str(list(sentiment_analysis)))
                    f.write('\n')

    def criteria_analysis(self, dict_classif, f_data, f_output):
        """Text sentiment analysis based on paper criteria"""

        with open(f_output + ".txt", "w") as f:
            pass

        with open(f_data + '.txt', 'r') as f:
            for line in f:
                try:
                    tweet = json.loads(line)
                    for item in dict_classif:
                        # Check if the user is not a probable spammer
                        # And if the user's author is the same as it is compared
                        if item[1] != "SPAM" and str(tweet['id']) == str(item[0]):
                            self.sentiment_analysis(item[0], tweet['text'], f_output)
                except:
                    pass
class Main(object):
    """Initializes script"""
    def start(self):
        """Runs spammer detection and sentiment analysis"""

        # --- Filenames' Variables ---
        # Each variable defines a filename
        raw_data = "paris__" + sys.argv[1]
        clean_data = "clean_" + sys.argv[1]
        # manual_classif_labels = "classification"
        manual_classif_clean_labels = "clean_cl_labels"
        # manual_classif_raw_data = "trained_data"
        manual_classif_clean_data = "clean_cl_data"
        spam_classification = "spam_classification" + sys.argv[1]
        sentiment_analysis = "sentiment_analysis" + sys.argv[1]
        # --- Pre Processing Data ---
        dpp = DataPreProcessing()
        # Get tweets manually classified from database and save them to file
        # dpp.get_manual_classified_tweets_from_database(manual_classif_raw_data)
        # Get classification given to tweets from database and save them to
        # file
        # dpp.get_manual_classification_from_database(manual_classif_labels)
        # Get all tweets from database and save them to file
        # dpp.get_all_tweets_from_database(raw_data)
        # Clean all tweets
        dpp.clean_data(raw_data, clean_data)
        # Clean tweets manually classified
        # dpp.clean_data(manual_classif_raw_data, manual_classif_clean_data)
        # Clean classification given to tweets
        # dpp.clean_data(manual_classif_labels, manual_classif_clean_labels,
        #               isJson=False)


        # --- Spam Detection ---
        sd = SpammerDetection()

        print("Gathering information from classified data")
        # with open(tweets_problem + '.txt', 'w+') as f:
        #     pass
        # Get all features from tweets
        features = sd.get_criteria_features_by_user(manual_classif_clean_data)
        # Makes list associating user id to manual classification
        classification = sd.manual_classification(manual_classif_clean_data,
                                                  manual_classif_clean_labels)

        ## Training Classifiers
        print("Training Classifiers")

        # Trains classifier based on Decision Tree method
        clf_dt = sd.trains_decision_tree(features, classification)
        # Trains classifier based on Naive Bayes Multinomial method
        clf_m = sd.trains_multinomial(features, classification)
        # Trains classifier based on Naive Bayes Bernoulli method
        clf_b = sd.trains_bernoulli(features, classification)

        # Probability of user being spammer based on features
        print("Calculating probability based on criteria")
        users_prob_spammer = sd.get_prob_based_on_criteria(features)
        # Remove duplicated users from list of probabilities above
        print("Removing duplicates")
        prob_spammer = sd.remove_duplicates_prob_criteria(users_prob_spammer)

        ## Classifying data for test purposes
        print("Classifying based on criteria")
        criteria = sd.criteria_classification(prob_spammer)
        print("Classifying based on Bernoulli")
        bernoulli = sd.classification(clf_b, manual_classif_clean_data)
        print("Classifying based on Decision Tree")
        decision_tree = sd.classification(clf_dt, manual_classif_clean_data)
        print("Classifying based on Multinomial")
        multinomial = sd.classification(clf_m, manual_classif_clean_data)

        ## 10-Fold Cross Validation of classifiers

        # Do Cross Validation 10-Fold to determine accuracy of Decision Tree
        # Classifier
        print("Decision Tree Classification ... ")
        accuracy_dt = sd.cross_validation_10_fold(clf_dt, features,
                                                  classification)
        # Do Cross Validation 10-Fold to determine accuracy of Multinomial
        # Classifier
        print("Multinomial Classification ... ")
        accuracy_m = sd.cross_validation_10_fold(clf_m, features,
                                                 classification)
        # Do Cross Validation 10-Fold to determine accuracy of Bernoulli
        # Classifier
        print("Bernoulli Classification ... ")
        accuracy_b = sd.cross_validation_10_fold(clf_b, features,
                                                 classification)

        print("10-Fold Cross Validation")
        print("Accuracy Bernoulli: {:.2f}".format(accuracy_b))
        print("Accuracy Multinomial: {:.2f}".format(accuracy_m))
        print("Accuracy Decision Tree: {:.2f}".format(accuracy_dt))
        with open(log_file, 'a') as log:
            log.write("10-Fold Cross Validation")
            log.write("\nAccuracy Bernoulli: " + str.format("{0:.2f}", accuracy_b))
            log.write("\nAccuracy Multinomial: " + str.format("{0:.2f}", accuracy_m))
            log.write("\nAccuracy Decision Tree: " + str.format("{:.2f}", accuracy_dt))
        sd.criteria_accuracy(criteria, classification)



        print("---- Confusion matrix ----")
        with open(log_file, 'a') as log:
            log.write("\n---- Confusion matrix ----")
        sd.confusion_matrix_(criteria, bernoulli, title="Bernoulli")
        sd.confusion_matrix_(criteria, multinomial, title="Multinomial")
        sd.confusion_matrix_(criteria, decision_tree, title="Decision Tree")
        print("--------------------------\n")
        with open(log_file, 'a') as log:
            log.write("\n--------------------------\n")

        # ---- Analysis of yet not classified data ----
        print("Analysis of yet not classified data")

        # Get all features from tweets
        print("Getting all features from users")
        features = sd.get_criteria_features_by_user(clean_data)
        # Probability of user being spammer based on features
        print("Calculating probability based on criteria")
        users_prob_spammer = sd.get_prob_based_on_criteria(features)
        # Remove duplicated users from list of probabilities above
        print("Removing duplicates")
        prob_spammer = sd.remove_duplicates_prob_criteria(users_prob_spammer)

        ## Classifying data
        print("Classifying based on criteria")
        criteria = sd.criteria_classification(prob_spammer)
        print("Classifying based on Bernoulli")
        bernoulli = sd.classification(clf_b, clean_data)
        print("Classifying based on Decision Tree")
        decision_tree = sd.classification(clf_dt, clean_data)
        print("Classifying based on Multinomial")
        multinomial = sd.classification(clf_m, clean_data)

        # Ver quantidade de spammers para cada metodo de classificacao


        # Tweets are classified as "SPAM" or "NOT SPAM"
        # Tweets classified as "NOT SPAM" returned as list
        print("Generating Final Classification")
        final_classif = sd.final_classification(criteria, bernoulli,
                                                decision_tree, multinomial)
        print("Saving Final Classification to file")
        with open(spam_classification + '.txt', 'a') as f:
            for item in final_classif:
                f.write(str(item) + '\n')

        # --- Sentiment Analysis ---
        # s_analysis = SentimentAnalysis()
        # Analyze sentiments of non-spam tweets and save it + user ID to file
        # s_analysis.criteria_analysis(final_classif, clean_data, sentiment_analysis)

        # --- Sample Characteristics ---
        characteristics = SampleCharacteristics()
        # Saves general data characteristics to file
        characteristics.display_general_characteristics(clean_data)
        # Save data characteristics' graphics to file
        # characteristics.plot_and_save_graphics(clean_data)

        characteristics.classification_results(bernoulli, title="Bernoulli")
        characteristics.classification_results(multinomial, title="Multinomial")
        characteristics.classification_results(decision_tree, title="Decision Tree")
        characteristics.classification_results(criteria, title="Criteria")
        characteristics.classification_results(final_classif, title="Final Classification")


if __name__ == '__main__':
    Main().start()
