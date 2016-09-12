# $ export COLOREDLOGS_LOG_FORMAT='%(asctime)s - %(message)s'

import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
# import coloredlogs
# coloredlogs.install(level='INFO')
# coloredlogs.install(level='DEBUG')
# coloredlogs.install(level='ERROR')

logging.info("Loading libraries ...\n")

try:
    logging.info("Started loading pdb")
    import pdb
    logging.debug("Pdb loaded successfully")
except Exception as ex:
    logging.error("Pdb loading failed")

try:
    logging.info("Started loading psycopg2")
    import psycopg2.extras
    logging.debug("Psycopg2 loaded successfully")

except Exception as ex:
    logging.error("Psycopg2 loading failed")

try:
    logging.info("Started loading nltk")
    import nltk
    logging.debug("Nltk loaded successfully")
except Exception as ex:
    logging.error("Nltk loading failed")

try:
    logging.info("Started loading sklearn.tree")
    from sklearn import tree
    logging.debug("Sklearn.tree loaded successfully")
except Exception as ex:
    logging.error("Sklearn.tree loading failed")

try:
    logging.info("Started loading sklearn.naive_bayes.BernoulliNB")
    from sklearn.naive_bayes import BernoulliNB
    logging.debug("Sklearn.naive_bayes.BernoulliNB loaded successfully")
except Exception as ex:
    logging.error("Sklearn.naive_bayes.BernoulliNB loading failed")

try:
    logging.info("Started loading sklearn.datasets")
    from sklearn import datasets
    logging.debug("Sklearn.datasets loaded successfully")
except Exception as ex:
    logging.error("Sklearn.datasets loading failed")

try:
    logging.info("Started loading numpy.array")
    from numpy import array
    logging.debug("Numpy.array loaded successfully")
except Exception as ex:
    logging.error("Numpy.array loading failed")

try:
    logging.info("Started loading StringIO.StringIO")
    from StringIO import StringIO
    logging.debug("StringIO loaded successfully")
except Exception as ex:
    logging.error("StringIO loading failed")

try:
    logging.info("Started loading matplotlib")
    import matplotlib
    matplotlib.use('Agg')
    logging.debug("Matplotlib loaded successfully")
except Exception as ex:
    logging.error("Matplotlib loading failed")

try:
    logging.info("Started loading matplotlib.pyplot")
    import matplotlib.pyplot as plt
    logging.debug("Matplotlib.pyplot loaded successfully")
except Exception as ex:
    logging.error("Matplotlib.pyplot loading failed")

try:
    logging.info("Started loading codecs")
    import codecs
    logging.debug("Codecs loaded successfully")
except Exception as ex:
    logging.error("Codecs loading failed")

try:
    logging.info("Started loading json")
    import json
    logging.debug("Json loaded successfully")
except Exception as ex:
    logging.error("Json loading failed")

try:
    logging.info("Started loading os")
    import os
    logging.debug("Os loaded successfully")
except Exception as ex:
    logging.error("Os loading failed")

logging.info("\nFinished loading libraries\n")

# try:
#     # Connection between Python and PostgreSQL
#     conn = psycopg2.connect()
#     # Create cursor
#     cur = conn.cursor()
#     # Execute the query
#     cur.execute("""SELECT json FROM paris limit 10000""")
#     # Save results to file
#     _data = codecs.open(os.path.expanduser('~/Desktop/data10000.txt'), 'w')
#     result = []
#     for row in cur.fetchall():
#         result.append(row[0])
#         _data.write(json.dumps(result[0]) + '\n')
#     _data.close()
#     # Close cursor and database connection
#     cur.close()
#     conn.close()
try:
    attributes_folder = os.path.expanduser('~/Attributes/')

    logging.info("Creating dictionaries")

    sentiment_analysis = {
        "user_id": "",
        "sentiment": set(),
    }

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
    logging.debug("Dicitonaries created successfully")
except Exception as ex:
    logging.error("Dictionaries creation failed")

try:
    logging.info("Loading dict of sentiments")
    # Dictionary of sentiments
    sentiments = {
        'fear/anxiety': ['anxiety', 'anxious', 'catastrophic', 'concern', 'disaster', 'emergency', 'fear', 'insecure', 'panic', 'scared', 'terror', 'threat', 'trouble', 'warning', 'worry'],
        'shock': ['taken aback', 'aback', 'floor', 'god bless', 'omg', 'shock', 'stun', 'sudden', 'wtf', 'wth'],
        'response': ['act', 'asap', 'escape', 'evacuate', 'flee', 'help', 'hide', 'run'],
        'need information': ['breaking news', 'call', 'foul play', 'incident', 'phone', 'report', 'situation', 'unconfirmed'],
        'threat': ['accident', 'attack', 'bomb', 'bullet', 'collapse', 'crash', 'explode', 'explosion', 'fire', 'gun', 'hijack', 'hit', 'hostage', 'plane', 'responsability', 'responsable', 'rifle', 'shoot', 'shot', 'struck', 'suicide', 'terrorism'],
        'casualities': ['blood', 'body', 'bodies', 'corpses', 'corpse', 'dead', 'injury', 'injure', 'kill', 'wounded'],
        'law enforcement': ['action', 'ambulance', 'command', 'medic', 'operation', 'planes', 'police', 'cops', 'FBI', 'security', 'recover', 'rescue', 'response', 'restore', 'safe', 'safety', 'save', 'shut', 'stay', 'survive', 'suspend'],
    }
    # End of dictionary
    logging.debug("Dict of sentiments loaded successfully")
except Exception as ex:
    logging.error("Dict of sentiments loaded failed")

try:
    logging.info("Creating charateristics files")
    # Creation of files
    File = open(attributes_folder + 'text.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'user_id.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'tweet_id.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'favorite_count.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'source.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'entities__user_mentions.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'entities__hashtags.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'user__followers_count.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'user__friends_count.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'geo.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'created_at.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'user__statuses_count.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'user__favourites_count.txt', 'w+')
    File.close()
    File = open(attributes_folder + 'user__listed_count.txt', 'w+')
    File.close()
    # End of file creation
    logging.debug("Files created successfully")
except Exception as ex:
    logging.error("Files creation failed")
    # Returns all values of a given attribute
def Get_Field_Values(field):
# Search for "attribute_folder + field + 'txt'" in the directory
    File = open(attributes_folder + field + '.txt', 'r')
    if File > 0:  # If file exists and is not blank
        attributes_list = []
        for line in File:
            attributes_list.append(line.rstrip('\n'))
            File.close()
            return attributes_list
    else:
        File.close()
        return []
# End of function
    
try:
    logging.info("Parsing tweets and writing to file")
    
    # Fill each attribute file with one attribute value per line
    with open(os.path.expanduser('~/json.txt'), 'r') as File:
        for line in File:
            # Parse tweet from file to JSON format
            try:
                line_object = json.loads(line)

                # Save total tweets to file
                File = codecs.open(attributes_folder + 'user__statuses_count.txt', 'a', encoding = 'utf-8')
                File.write(str(line_object['user']['statuses_count']) + '\n')
                File.close()

                # Save number of times the user has been added to
                # someone else's list
                File = codecs.open(attributes_folder + 'user__listed_count.txt', 'a', encoding = 'utf-8')
                File.write(str(line_object['user']['listed_count']) + '\n')
                File.close()

                # Save to file number of times the user has been added to someone else's favorite list
                File = codecs.open(attributes_folder + 'user__favourites_count.txt', 'a', encoding = 'utf-8')
                File.write(str(line_object['user']['favourites_count']) + '\n')
                File.close()

                # Save tweet content to file
                File = codecs.open(attributes_folder + 'text.txt', 'a', encoding = 'utf-8')
                if line_object['text'] is not None:
                    File.write(line_object['text'])
                File.write('\n')
                File.close()

                # Save user id to file
                File = codecs.open(attributes_folder + 'user_id.txt', 'a', encoding = 'utf-8')
                File.write(str(line_object['id']) + '\n' if line_object['id'] is not None else '\n')
                File.close()

                # Save tweet id to file
                File = codecs.open(attributes_folder + 'tweet_id.txt', 'a', encoding = 'utf-8')
                File.write(str(line_object['timestamp_ms']) + '\n')
                File.close()

                # Save to file how many times user has been favorited
                File = codecs.open(attributes_folder + 'favorite_count.txt', 'a', encoding='utf-8')
                File.write(str(line_object['favorite_count']) + '\n' if line_object['favorite_count'] is not None else '\n')
                File.close()

                # Save to file channel used by user to tweet
                File = codecs.open(attributes_folder + 'source.txt', 'a', encoding = 'utf-8')
                File.write(line_object['source'] + '\n' if line_object['source'] is not None else '\n')
                File.close()

                # Save to file total of user mentions
                File = codecs.open(attributes_folder + 'entities__user_mentions.txt', 'a', encoding = 'utf-8')
                File.write(str(line_object['entities']['user_mentions'][0]['id']) + '\n' if len(line_object['entities']['user_mentions']) > 0 else '\n')
                File.close()

                # Save to file hashtags used in the tweet
                File = codecs.open(attributes_folder + 'entities__hashtags.txt', 'a', encoding = 'utf-8')
                if len(line_object['entities']['hashtags']) > 0:
                    File.write(str(line_object['entities']['hashtags']))
                File.write('\n')
                File.close()

                # Save to file user's followers
                File = codecs.open(attributes_folder + 'user__followers_count.txt', 'a', encoding='utf-8')
                File.write(str(line_object['user']['followers_count']) + '\n')
                File.close()

                # Save to file user's friends/following count
                File = codecs.open(attributes_folder + 'user__friends_count.txt', 'a', encoding='utf-8')
                File.write(str(line_object['user']['friends_count']) + '\n')
                File.close()

                # Save to file tweet's geolocation information
                File = codecs.open(attributes_folder + 'geo.txt', 'a', encoding= 'utf-8')
                if line_object['geo'] is not None:
                    File.write(str(line_object['geo']))
                File.write('\n')
                File.close()

                # Save to file date/time when tweet was created
                File = codecs.open(attributes_folder + 'created_at.txt', 'a', encoding='utf-8')
                if len(line_object['created_at']) > 0:
                    File.write(str(line_object["created_at"]))
                File.write('\n')
                File.close()

            except:
                pass
    # End of file filling

    logging.debug("Parsing and files' writing completed successfully")
except Exception as ex:
    logging.error("Parsing and files' writing completed failed")

try:
    logging.info("Specifying date and time ranges")
    
    # Create lists of all date/time
    dates = []
    times = []

    for item in Get_Field_Values('created_at'):
        month = item[4:7]
        day = item[8:10]
        year = item[26:30]
        hour = item[11:13]
        minutes = item[14:16]
        seconds = item[17:19]
        dates.append(day + '/' + month + '/' + year)
        times.append(hour + ':' + minutes + ':' + seconds)
    
    logging.debug("Data and time specified successfully")
except Exception as ex:
    logging.error("Data and time specified failed")
    
try:
    logging.info("\n ----- SAMPLE CHARACTERISTICS ----- \n")

    # Count number of tweets
    tweets_count = len(Get_Field_Values('tweet_id'))
    print "Number of tweets: %d" % tweets_count

    # Count number of unique users
    unique_users = len(set(Get_Field_Values('user_id')))
    print "Number of unique users: %d" % unique_users

    # Average of tweets per user with graphic
    average_tweets_user = tweets_count / unique_users if unique_users != 0  else 0
    print "Average of tweets per user: %s" % average_tweets_user

    # Date/time range
    print "Date range: %s to %s" % (min(dates), max(dates))
    print "Time range: %s to %s" % (min(times), max(times))

    tweets_per_user = list((map(int, Get_Field_Values('user_id')).count(x)) for x in map(int, Get_Field_Values('user_id')))
    users_tweeted = list((tweets_per_user.count(x)) for x in set(tweets_per_user))
    times_tweeted = list(set(tweets_per_user))
except Exception as ex:
    logging.error("Error displaying results")

logging.info("\n ---------------- \n")

try:
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
    logging.info("\nAnalyzing sample data...\n")
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
        'follows more than double than being followd': 0.5,
        'doesnt have favourite list': 0.17,
    }

    dict_threshold_0_991 = {}
    dict_threshold_0_99 = {}
    dict_threshold_0_9 = {}
    criteria_prob = []
    dict_classificacao_manual = {}

    # Goes through each tweet and assess if it is probably from a spammer
    matrix_user_features_all_users = []
    with open(attributes_folder + 'user__favourites_count.txt', 'r') as file_user_listed_as_favorite, open(attributes_folder + 'user__listed_count.txt', 'r') as file_user_listed_by_another_user, open(attributes_folder + 'user__statuses_count.txt' ,'r') as file_user_total_tweets, open(attributes_folder + 'user__followers_count.txt', 'r') as file_user_followers, open(attributes_folder + 'user__friends_count.txt', 'r') as file_user_friends_following, open(attributes_folder + 'user_id.txt', 'r') as file_user_id, open(attributes_folder + 'geo.txt', 'r') as file_user_geo, open(attributes_folder + 'favorite_count.txt', 'r') as file_user_favorited, open(attributes_folder + 'entities__hashtags.txt', 'r') as file_tweet_hashtags, open(attributes_folder + 'text.txt', 'r') as file_tweet_text, open(attributes_folder + 'entities__user_mentions.txt', 'r') as file_user_mentions, open(attributes_folder + "source.txt", 'r') as file_source, open('/home/amirelemam/classificacao.txt', 'r') as file_classificacao:
        matrix_user_id_vs_spammer_criteria = []
        for user_favorited, user_added_in_list, user_total_tweets, user_id, user_followers, user_following, user_geo, user_favorited, tweet_hashtags, tweet_text, user_mentions, user_source, classificacao in zip(file_user_listed_as_favorite, file_user_listed_by_another_user, file_user_total_tweets, file_user_id, file_user_followers, file_user_friends_following, file_user_geo, file_user_favorited, file_tweet_hashtags, file_tweet_text, file_user_mentions, file_source, file_classificacao):
            matrix_user_features = []
            prob_total = 1

            # Number of followers > 30 Weight: 0.53
            if int(user_following.rstrip('\n')) < 30:
                prob_total *= 0.53
                matrix_user_features.append(1)
            else:
                matrix_user_features.append(0)

            # Geolocation == true   Weight: 0.85
            if user_geo == "":
                prob_total *= 0.85
                matrix_user_features.append(1)
            else:
                matrix_user_features.append(0)

            # User included in another user's favorite  Weight: 0.85
            if user_favorited == 0:
                prob_total *= 0.85
                matrix_user_features.append(1)
            else:
                matrix_user_features.append(0)

            # It has used a hashtag at least once   Weight: 0.96
            if tweet_hashtags == '':
                prob_total *= 0.96
                matrix_user_features.append(1)
            else:
                matrix_user_features.append(0)

            # Logged in on an iPhone    Weight: 0.917
            if "iPhone" not in user_source:
                prob_total *= 0.917
                matrix_user_features.append(1)
            else:
                matrix_user_features.append(0)

            # Mentioned by another user Weight: 1
            if user_mentions == "":
                prob_total *= 1
                matrix_user_features.append(1)
            else:
                matrix_user_features.append(0)

            # User has less than 50 tweets  Weight: 0.01
            if user_total_tweets > 50:
                prob_total *= 0.01
                matrix_user_features.append(1)
            else:
                matrix_user_features.append(0)

            # User has been included in another user's list Weight: 0.45
            if user_added_in_list == 0:
                prob_total *= 0.45
                matrix_user_features.append(1)
            else:
                matrix_user_features.append(0)

            # Number of following is 2x or less Number of followers Weight: 0.5
            if int(user_followers)*2 < int(user_following):
                prob_total *= 0.5
                matrix_user_features.append(1)
            else:
                matrix_user_features.append(0)

            matrix_user_features_all_users.append(matrix_user_features)
            # User has at least one favorite list   Weight: 0.17
            criteria_prob.append(prob_total)

            # Add probability of being a spammer to a dictionary
            if 1 - prob_total > 0.5: dict_probable_spammers[str(user_id)] = prob_total

            # Test false positives and true positives
            if classificacao.strip() == "SPAM":
                dict_classificacao_manual[str(user_id)] = 1
            else:
                dict_classificacao_manual[str(user_id)] = 0

            if 1 - prob_total > 0.991:
                if classificacao.strip() == "SPAM":
                    dict_threshold_0_991[str(user_id)] = 1
                else:
                    dict_threshold_0_991[str(user_id)] = 0

            if 1 - prob_total > 0.99:
                if classificacao.strip() == "SPAM":
                    dict_threshold_0_99[str(user_id)] = 1
                else:
                    dict_threshold_0_99[str(user_id)] = 0

            if 1 - prob_total > 0.9:
                if classificacao.strip() == "SPAM":
                    dict_threshold_0_9[str(user_id)] = 1
                else:
                    dict_threshold_0_9[str(user_id)] = 0

            # Adds probability of being a spammer to a dictionary

            # Loops creates full data matrix
            if classificacao.strip() == "SPAM":
                class_labels.append(1)
                class_labels_SPAM_NOTSPAM.append("SPAM")
            else:
                class_labels.append(0)
                class_labels_SPAM_NOTSPAM.append("NOT SPAM")

            # matrix_data.append([sum(matrix_user_features)])
            matrix_data.append([prob_total])

    logging.debug("Sample data analyzed")
except Exception as ex:
    logging.error("Error analyzing sample data")

try:
    logging.info("Training classifier...\n")
    

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
    logging.info("\n --- CLASSIFICATION RESULTS --- \n")

    with open(attributes_folder + 'CriteriaClassification_vs_ManualClassification.txt', 'w+') as f:
        f.write("Criteria\tManual\tBernoulli\tDecision Tree\n")
    print("\nCriteria\tManual\tBernoulli\tDecision Tree")

    for criteria, classification, features in zip(matrix_data, class_labels_SPAM_NOTSPAM, matrix_user_features_all_users):
        with open(attributes_folder + 'CriteriaClassification_vs_ManualClassification.txt', 'a') as f:
            f.write("%f\t%s\t%s\t%s\n" % (criteria[0], classification, clf_bernoulli.predict(array(features))[0], clf_binary_decision_tree.predict(array(sum(list(features))))[0]))
        print "%f\t%s\t%s\t%s" % (criteria[0], classification, clf_bernoulli.predict(array(features))[0], clf_binary_decision_tree.predict(array(sum(list(features))))[0])

    print("\nProbability for a SURELY NOT SPAM, according to Bernoulli classification:")
    if clf_bernoulli.predict(array([0, 0, 0, 0, 0, 0, 0, 0, 0])) == 1:
        print("SPAM")
    else:
        print("NOT SPAM")

except Exception as ex:
    logging.error("Error displaying classification results")

try:
    logging.info("Exporting trained data to file")
    # Export trained data to file
    with open(attributes_folder + 'trained_data_binary_decision_tree.dot', 'w') as f:
        f = tree.export_graphviz(clf_binary_decision_tree, out_file=f)

#     with open(attributes_folder + 'trained_data_bernoulli.dot', 'w') as f:
#         f = tree.export_graphviz(clf_bernoulli, out_file=f)

    logging.debug("Trained data saved to file")
except Exception as ex:
    logging.error("Error saving trained data to file")

try:
    logging.info("Detecting spammers... ")
    # Determinate if the probable spammer is above threshold
    # And count the total number of probable spammers
    totalProbableSpammers = 0
    File = open(attributes_folder + "probable_spammers.txt", "w+")
    for probableSpammer in dict_probable_spammers.keys():
#        if dict_probable_spammers[probableSpammer] > 0.1:
        s = "{%s: %s}\n" % (probableSpammer,  dict_probable_spammers[probableSpammer])
        File.write(str(s))
        totalProbableSpammers += 1
    File.close()

    # Prints the total number of probable spammers
    print "\nTotal Probable Spammers: %d\n" % totalProbableSpammers

    print "Over 0.991: %d users" % len(dict_threshold_0_991.keys())
    count = 0
    for item in dict_threshold_0_991.keys():
        count += dict_threshold_0_991[item]
    print "Over 0.991 True positives: %d\n" % count

    print "Over 0.99: %d users" % len(dict_threshold_0_99.keys())
    count = 0
    for item in dict_threshold_0_99.keys():
        count += dict_threshold_0_99[item]
    print "Over 0.99 True positives: %d\n" % count

    print "Over 0.9: %d users" % len(dict_threshold_0_9.keys())
    count = 0
    for item in dict_threshold_0_9.keys():
        count += dict_threshold_0_9[item]
    print "Over 0.9 True positives: %d\n" % count
    
except Exception as ex:
    logging.error("Error detecting spammers")

try:
    logging.info("Analyzing sentiments...")

    # Does the sentiment analysis
    # Writes the sentiment fo each user to a file
    File = open(attributes_folder + "sentiment_analysis.txt", "w+")
    with open(attributes_folder + "text.txt", "r") as file_tweet_text, open(attributes_folder + "user_id.txt", "r") as file_user_id, open(attributes_folder + "tweet_id.txt", "r") as file_tweet_id:
        for tweet_text, user_id, tweet_id in zip(file_tweet_text, file_user_id, file_tweet_id):
            for key in sentiments.keys():
                for i in range(len(sentiments[key])):
                    if sentiments[key][i] in tweet_text:
                        sentiment_analysis["user_id"] = str(user_id.strip("\n"))
                        sentiment_analysis["sentiment"].add(key)

            for item in sentiment_analysis:
                if ", ".join(sentiment_analysis["sentiment"]) != "":
                    s = "{\"user_id:\" %s, \"tweet_id\": %s, \"sentiments\": %s}\n" % (sentiment_analysis["user_id"], tweet_id, list(sentiment_analysis["sentiment"]))
                    File.write(str(s))
    File.close()
    logging.debug("Sentiments analyzed and saved to file")
except Exception as ex:
    logging.error("Error analyzing sentiments or saving it to file")
    print ex
