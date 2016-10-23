from __future__ import division

import json
import logging
import pdb
# from sklearn.model_selection import train_test_split
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import psycopg2.extras
from numpy import array
from sklearn import datasets, tree
from sklearn.naive_bayes import BernoulliNB

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
matplotlib.use('Agg')
# try:
#     # Connection between Python and PostgreSQL
#     conn = psycopg2.connect()
#     # Create cursor
#     cur = conn.cursor()
#     # Execute the query - Download trained data from database
#     cur.execute("""\COPY (select p.json from paris_analise pa inner join
#     paris p on p.codtweet = pa.codtweet) TO 'trained_data.txt'""")
#     # Close cursor and database connection
#     cur.close()
#     cur = conn.cursor()
#     # Execute the query - Download classified data from database
#     cur.execute("""\COPY (select pa.classificacao from paris_analise pa inner join paris p on p.codtweet = pa.codtweet) TO '/Users/amirelemam/Desktop/classificacao.txt'""")
#     cur.close()
#     cur = conn.cursor()
#     # Execute the query - Download all data from database
#     cur.execute("""SELECT json FROM paris""")
#     # Save results to file
#     _data = codecs.open(os.path.expanduser('data.txt'), 'w')
#     for row in cur.fetchall():
#         _data.write(json.dumps(row[0]) + '\n')
#     _data.close()
#     # Close cursor and database connection
#     cur.close()
#     conn.close()
# except:
#     logging.error("Error reading trained data from database")
# 
# with open("~/trained_data.txt", "r") as s, open("~/clean_data.txt", "w+") as dest:
#     for line in s:
#         line = line.replace('\\\\"', "@")
#         dest.write(str(line))

with open('sentiment_analysis.txt', 'w+') as f:
    f.write("Sentiment Analysis\n")

sentiment_analysis = {
    "user_id": "",
    "sentiment": set(),
}

manual_classification = []
bernoulli_classification = []
decision_tree_classification = []
sample_features = []

# Dictionary of months
months = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12',
}
# End of dictionary

# Dictionary of sentiments
sentiments = {
    'fear/anxiety': ['anxiety', 'anxious', 'catastrophic', 'concern',
        'disaster', 'emergency', 'fear', 'insecure', 'panic',
        'scared', 'terror', 'threat', 'trouble', 'warning', 'worry'],
    'shock': ['taken aback', 'aback', 'floor', 'god bless', 'omg',
        'shock', 'stun', 'sudden', 'wtf', 'wth'],
    'response': ['act', 'asap', 'escape', 'evacuate', 'flee', 'help',
        'hide', 'run'],
    'need information': ['breaking news', 'call', 'foul play',
        'incident', 'phone', 'report', 'situation', 'unconfirmed'],
    'threat': ['accident', 'attack', 'bomb', 'bullet', 'collapse',
        'crash', 'explode', 'explosion', 'fire', 'gun', 'hijack',
        'hit', 'hostage', 'plane', 'responsability', 'responsable',
        'rifle', 'shoot', 'shot', 'struck', 'suicide', 'terrorism'],
    'casualities': ['blood', 'body', 'bodies', 'corpses', 'corpse',
        'dead', 'injury', 'injure', 'kill', 'wounded'],
    'law enforcement': ['action', 'ambulance', 'command', 'medic',
        'operation', 'planes', 'police', 'cops', 'FBI', 'security',
        'recover', 'rescue', 'response', 'restore', 'safe', 'safety',
        'save', 'shut', 'stay', 'survive', 'suspend'],
}
# End of dictionary

try:
    logging.info("\nAnalyzing sample data...")
    dict_probable_spammers = {}
    source = set()
    probable_spammer = []

    # Matrices for DataTreeClassifier
    # DecisionTreeClassifier(X, Y, class_weight)
    # X: matrix[samples, features] for training
    # Y: class labels
    # class_weight: dict of weight {class_label: weight}
    matrix_data = []
    class_labels = []
    class_labels_SPAM_NOTSPAM = []
    class_weight_dict = {
        'less than 30 followers': 0.53,
        'not geolocalized': 0.83,
        'not included in another users favourite': 0.85,
        'havent used hashtags': 0.96,
        'didnt log in on iPhone': 0.917,
        'not mentioned by another user': 1,
        'wrote less than 50 tweets': 0.01,
        'not included in another users list': 0.45,
        'follows more than double than being followed': 0.5,
        'doesnt have favourite list': 0.17,
    }

    criteria_prob = []
    classificacao_manual = [] 
    count_tweets = 0
    total_unique_users = [] 

    # Create lists of all date/time
    dates = []
    times = []

    matrix_user_features_all_users = []
    matrix_user_id_vs_spammer_criteria = []
    count = 0
except:
    pass

try:
    logging.info("Detecting spammers and analyzing sentiments")
    
    # Fill each attribute file with one attribute value per line
    with open('clean_data.txt', 'r') as data, open('/home/amirelemam/classificacao.txt', 'r') as Classification:
        try:
            for line, classification in zip(data, Classification):
                # Parse tweet from file to JSON format
                line_object = json.loads(line)
                # Goes through each tweet and assess if it is probably from a spammer
                user_features = []
                prob_total = 1
                following = int(line_object['user']['friends_count'])
                followers = int(line_object['user']['followers_count'])  
                user_id = str(line_object['id'])
                total_unique_users.append(int(user_id))
                
                # Number of followers > 30 Weight: 0.53
                # User's friends/following count
                if following < 30:
                    prob_total *= 0.53
                    user_features.append(1)
                else:
                    user_features.append(0)

                # Geolocation == true   Weight: 0.85
                # Tweet's geolocation information
                if line_object['geo'] is not None:
                    prob_total *= 0.85
                    user_features.append(1)
                else:
                    user_features.append(0)

                # User included in another user's favorite  Weight: 0.85
                # Save to file number of times the user has been added to 
                # someone else's favorite list
                if int(line_object['user']['favourites_count']) == 0:
                    prob_total *= 0.85
                    user_features.append(1)
                else:
                    user_features.append(0)

                # It has used a hashtag at least once   Weight: 0.96
                # Hashtags used in the tweet
                if len(line_object['entities']['hashtags']) > 0:
                    prob_total *= 0.96
                    user_features.append(1)
                else:
                    user_features.append(0)

                # Logged in on an iPhone    Weight: 0.917
                # Channel used by user to tweet
                if line_object['source'] is not None:
                    if "iPhone" not in line_object['source']: 
                        prob_total *= 0.917
                        user_features.append(1)
                    else:
                        user_features.append(0)
                else:
                    user_features.append(0)

                # Mentioned by another user Weight: 1
                # Total of user mentions
                if len(line_object['entities']['user_mentions']) > 0:
                    prob_total *= 1
                    user_features.append(1)
                else:
                    user_features.append(0)

                # User has less than 50 tweets  Weight: 0.01
                # Save total tweets to file
                if int(line_object['user']['statuses_count']) > 50:
                    prob_total *= 0.01
                    user_features.append(1)
                else:
                    user_features.append(0)

                # User has been included in another user's list Weight: 0.45
                # Save number of times the user has been added to
                # someone else's list
                if int(line_object['user']['listed_count']) == 0:
                    prob_total *= 0.45
                    user_features.append(1)
                else:
                    user_features.append(0)

                # Number of following is 2x or less Number of followers Weight: 0.5
                if followers*2 < following:
                    prob_total *= 0.5
                    user_features.append(1)
                else:
                    user_features.append(0)
                # User has at least one favorite list   Weight: 0.17

                # Create list (user register) with user id and list of features
                user_id_and_features.append(deepcopy(user_id))
                user_id_and_features.append(deepcopy(user_features))
                # Insert previous list (user register) in general list
                sample_features.append(deepcopy(user_id_and_features))

                # Create list (user register) with user id and prob of being
                # spammer 
                user_id_and_prob_spammer.append(deepcopy(user_id))
                user_id_and_prob_spammer.append(deepcopy(prob_total))
                # Insert previous list (user register) in general list
                criteria_classification.append(deepcopy(user_id_and_prob_spammer))

                # Create list (user register) with user id and manual
                # classification
                user_id_and_manual_cl.append(deepcopy(user_id))
                user_id_and_manual_cl.append(deepcopy(classification))
                # Insert previous list (user register) in general list
                manual_classification.append(deepcopy(user_id_and_manual_cl))
                
               
                item = str(line_object["created_at"])
                month = item[4:7]
                day = item[8:10]
                year = item[26:30]
                hour = item[11:13]
                minutes = item[14:16]
                seconds = item[17:19]
                dates.append(day + '/' + month + '/' + year)
                times.append(hour + ':' + minutes + ':' + seconds)
                    
                # Does the sentiment analysis
                # Writes the sentiment fo each user to a file
                tweet_text = str(line_object['text']) 
                tweet_id = str(line_object['timestamp_ms']) 
                for key in sentiments.keys():
                    for i in range(len(sentiments[key])):
                        if sentiments[key][i] in tweet_text:
                            sentiment_analysis["user_id"] = user_id
                            sentiment_analysis["sentiment"].add(key)

                with open('sentiment_analysis.txt', 'a') as sentiment_file:
                    for item in sentiment_analysis:
                        sentiment_file.write(item)

                # Test false positives and true positives
                # Loops creates full data matrix
                classificacao_manual.append([user_id,
                    classification.strip()])
                if classification.strip() == "SPAM":
                    class_labels.append((a[1] if a[1] for a
                        in classificacao_manual))
                    class_labels_SPAM_NOTSPAM = (a[1] for a
                        in classificacao_manual)

                count_tweets += 1
            
        except:
            pass

    logging.debug("Spammers detected and sentiments analyzed successfully")
except Exception as ex:
    logging.error("Spammers detection and sentiments analysis failed")

    print count_tweets

try:
    logging.info("\n ----- SAMPLE CHARACTERISTICS ----- \n")

    # Count number of tweets
    print "Number of tweets: %d" % count_tweets 

    # Count number of unique users
    print "Number of unique users: %d" % len(set(total_unique_users))

    # Average of tweets per user with graphic
    if len(set(total_unique_users)):
        print "Average of tweets per user: %s" % (count_tweets /
               len(set(total_unique_users)))
    else:
        print "Average of tweets per user: 0"

    # Date/time range
    print "Date range: %s to %s" % (min(dates), max(dates))
    print "Time range: %s to %s" % (min(times), max(times))

except Exception as ex:
    logging.error("Error displaying results")

logging.info("\n ---------------- \n")

try:
    tweets_per_user = list((map(int, total_unique_users).count(x)) for x in
            map(int, total_unique_users))
    users_tweeted = list((tweets_per_user.count(x)) for x in set(tweets_per_user))
    times_tweeted = list(set(tweets_per_user))

    logging.info("Saving graphic \'Qty of tweets per user\' to file")

    # Plot bar graphic of tweets/user
    plt.bar(times_tweeted, users_tweeted, width=1, color="blue")
    plt.xlabel('# of tweets')
    plt.ylabel('Qty users tweeted')
    plt.title('# of tweets per user')
    plt.savefig("tweets_per_user.jpg")

    logging.debug("Graphic \'Qty of tweets per user\' successfully saved to file")
except Exception as ex:
    logging.error("Failed to save graphic \'Qty of tweets per user\' to file")

try:
    logging.info("Saving graphic \'Qty of tweets per day\' to file")
    # Graphic of tweets/day
    # Count number of tweets/day
    dict_tweet_count = dict((x, dates.count(x)) for x in set(dates))
    list_tweet_count = list((dates.count(x)) for x in set(dates))

    # Plot bar graphic of tweets/day
    x = range(len(list_tweet_count))
    plt.bar(x, list_tweet_count, width=1, color="blue")
    plt.xlabel('Day')
    plt.ylabel('# of tweets')
    plt.title('# of tweets per day')
    plt.savefig("tweets_per_day.jpg")
    # plt.show()
    logging.debug("Graphic \'Qty of tweets per day\' successfully saved to file")
except Exception as ex:
    logging.error("Failed to save graphic \'Qty of tweets per day\' to file")

try:
    logging.info("\nTraining classifier...")

    # Binary Decision Tree
    clf_binary_decision_tree = tree.DecisionTreeClassifier()
    clf_binary_decision_tree = clf_binary_decision_tree.fit(array(matrix_data), array(class_labels_SPAM_NOTSPAM))
    # clf_binary_decision_tree = clf_binary_decision_tree.fit(matrix_data, class_labels, class_weight_dict)

    # Bernoulli
    clf_bernoulli = BernoulliNB()
    clf_bernoulli.fit(array(matrix_user_features_all_users), array(class_labels_SPAM_NOTSPAM))

    logging.debug("Classifiers trained")
except Exception as ex:
    logging.error("Error training classifier")

try:
    logging.info("\n --- CLASSIFICATION RESULTS --- ")

    spam_spam_spam = 0
    spam_not_spam_spam = 0
    spam_spam_not_spam = 0
    not_spam_spam_spam = 0
    not_spam_not_spam_not_spam = 0
    not_spam_not_spam_spam = 0
    not_spam_spam_not_spam = 0
    spam_not_spam_not_spam = 0

    manual_classification_spammer = 0
    manual_classification_not_spammer = 0

    bernoulli_classification_spammer = 0
    bernoulli_classification_not_spammer = 0
    
    decision_tree_classification_spammer = 0
    decision_tree_classification_not_spammer = 0


    for criteria, classification, features in zip(matrix_data, class_labels_SPAM_NOTSPAM, matrix_user_features_all_users):
        with open('CriteriaClassification_vs_ManualClassification.txt', 'a') as f:
            f.write("%f\t%s\t%s\t%s\n" % (criteria[0], classification, clf_bernoulli.predict(array(features))[0], clf_binary_decision_tree.predict(array(sum(list(features))))[0]))
        if classification == 'SPAM':
            manual_classification_spammer += 1
            if clf_bernoulli.predict(array(features))[0] == 'SPAM':
                bernoulli_classification_spammer += 1
                if clf_binary_decision_tree.predict(array(sum(list(features))))[0] == 'SPAM':
                    spam_spam_spam += 1
                    decision_tree_classification_spammer += 1
                else:
                    spam_spam_not_spam += 1
                    decision_tree_classification_not_spammer += 1
            else:
                bernoulli_classification_not_spammer += 1
                if clf_binary_decision_tree.predict(array(sum(list(features))))[0] == 'SPAM':
                    spam_not_spam_spam += 1
                    decision_tree_classification_spammer += 1
                else:
                    spam_not_spam_not_spam += 1
                    decision_tree_classification_not_spammer += 1
        else:
            manual_classification_not_spammer += 1
            if clf_bernoulli.predict(array(features))[0] == 'SPAM':
                bernoulli_classification_spammer += 1
                if clf_binary_decision_tree.predict(array(sum(list(features))))[0] == 'SPAM':
                    not_spam_spam_spam += 1
                    decision_tree_classification_spammer += 1
                else:
                    not_spam_spam_not_spam += 1
                    decision_tree_classification_not_spammer += 1
            else:
                bernoulli_classification_not_spammer += 1 
                if clf_binary_decision_tree.predict(array(sum(list(features))))[0] == 'SPAM':
                    not_spam_not_spam_spam += 1
                    decision_tree_classification_spammer += 1
                else:
                    not_spam_not_spam_not_spam += 1
                    decision_tree_classification_not_spammer += 1

    print("MANUAL\t\tBERNOULLI\tDECISION TREE\tTOTAL")
    print("SPAM\t\tSPAM\t\tSPAM\t\t{}".format(spam_spam_spam))
    print("SPAM\t\tSPAM\t\tNOT SPAM\t{}".format(spam_spam_not_spam))
    print("SPAM\t\tNOT SPAM\tSPAM\t\t{}".format(spam_not_spam_spam))
    print("NOT SPAM\tSPAM\t\tSPAM\t\t{}".format(not_spam_spam_spam))
    print("NOT SPAM\tNOT SPAM\tNOT SPAM\t{}".format(not_spam_not_spam_not_spam))
    print("NOT SPAM\tNOT SPAM\tSPAM\t\t{}".format(not_spam_not_spam_spam))
    print("SPAM\t\tNOT SPAM\tNOT SPAM\t{}".format(spam_not_spam_not_spam))
    print("NOT SPAM\tSPAM\t\tNOT SPAM\t{}".format(not_spam_spam_not_spam))

    if count_tweets > 0:
        bernoulli_true_positive = (spam_spam_spam + spam_spam_not_spam) / count_tweets 
        bernoulli_true_negative = (not_spam_not_spam_not_spam +
                not_spam_not_spam_spam) / count_tweets 
        bernoulli_false_positive = (not_spam_spam_spam +
                not_spam_spam_not_spam) / count_tweets 
        bernoulli_false_negative = (spam_not_spam_not_spam +
                spam_not_spam_spam) / count_tweets 
        decision_tree_true_positive = (spam_spam_spam + spam_not_spam_spam) / count_tweets 
        decision_tree_true_negative = (not_spam_not_spam_not_spam +
                not_spam_spam_not_spam) / count_tweets 
        decision_tree_false_positive = (not_spam_not_spam_spam +
                not_spam_spam_spam) / count_tweets
        decision_tree_false_negative = (spam_spam_not_spam +
                spam_not_spam_not_spam) / count_tweets 
    else:
        bernoulli_true_positive = 0.0
        bernoulli_true_negative = 0.0
        bernoulli_false_positive = 0.0
        bernoulli_false_negative = 0.0
        decision_tree_true_positive = 0.0
        decision_tree_true_negative = 0.0
        decision_tree_false_positive = 0.0
        decision_tree_false_negative = 0.0

    print("\nBernoulli")
    print("True Positive: {0:.1f}%".format(bernoulli_true_positive*100))
    print("True Negativw: {0:.1f}%".format(bernoulli_true_negative*100))
    print("False Positive: {0:.1f}%".format(bernoulli_false_positive*100))
    print("False Negative: {0:.1f}%".format(bernoulli_false_negative*100))

    print("\nDecision Tree")
    print("True Positive: {0:.1f}%".format(decision_tree_true_positive*100))
    print("True Negativw: {0:.1f}%".format(decision_tree_true_negative*100))
    print("False Positive: {0:.1f}%".format(decision_tree_false_positive*100))
    print("False Negative: {0:.1f}%".format(decision_tree_false_negative*100))

    criteria_not_spammer = 0
    criteria_spammer = 0
    for criteria in matrix_data:
        if criteria[0] > 0.009:
            criteria_spammer += 1
        else:
            criteria_not_spammer += 1

    print("\nManual")
    print("Total spammers: {}".format(manual_classification_spammer))
    print("Total not spammers: {}".format(manual_classification_not_spammer))

    print("\nCriteria")
    print("Total spammers: {}".format(criteria_spammer))
    print("Total not spammers: {}".format(criteria_not_spammer))

    print("\nBernoulli")
    print("Total spammers: {}".format(bernoulli_classification_spammer))
    print("Total not spammers: {}".format(bernoulli_classification_not_spammer))

    print("\nDecision Tree")
    print("Total spammers: {}".format(decision_tree_classification_spammer))
    print("Total not spammers: {}".format(decision_tree_classification_not_spammer))

    print("\n")
    
except Exception as ex:
    logging.error("Error displaying classification results")

try:
    logging.info("Exporting trained data to file")
    # Export trained data to file
    with open('trained_data_binary_decision_tree.dot', 'w') as f:
        f = tree.export_graphviz(clf_binary_decision_tree, out_file=f)

    logging.debug("Trained data saved to file")
except Exception as ex:
    logging.error("Error saving trained data to file")

try:
    logging.info("\nDetecting probable spammers... ")
    # Determinate if the probable spammer is above threshold
    # And count the total number of probable spammers
#   open("probable_spammers.txt", "w+")

    # Prints the total number of probable spammers
    print "\nTotal Probable Spammers: %d\n" % totalProbableSpammers

except Exception as ex:
    print ex