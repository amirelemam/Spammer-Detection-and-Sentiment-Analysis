from __future__ import division

count = 0
text = ""
fields = {}

files = ['log_15-11-2016_3 46 43.txt', 'log_15-11-2016_5 49 50.txt', 'log_15-11-2016_4 0 54.txt', 'log_15-11-2016_5 9 16.txt', 'log_15-11-2016_4 13 23.txt', 'log_15-11-2016_6 15 14.txt', 'log_15-11-2016_4 28 7.txt', 'log_15-11-2016_6 2 46.txt', 'log_15-11-2016_4 41 30.txt', 'log_15-11-2016_6 28 14.txt', 'log_15-11-2016_4 55 28.txt', 'log_15-11-2016_6 40 24.txt', 'log_15-11-2016_5 23 5.txt', 'log_15-11-2016_6 52 18.txt' ,'log_15-11-2016_5 36 49.txt']

for index in [29, 30, 35, 36, 38, 39, 41, 42, 44, 45, 47, 48]:
    fields[index] = 0

for item in files:
    count += 0
    with open(item,"r") as f:
        for line in f:
            if count in [29, 30]:
                fields[count] += int(line.split(":")[1].strip())
            elif count in fields.keys():
                fields[count] += int(line.split(":")[1].strip().split(" ")[0])
            else:
                pass
            count += 1

lines = 0
rows = text.split("\n")
for i in range(count):
    if i == 29:
        number_tweets = fields[i]
        print "Number of tweets:", fields[i]
    elif i == 30:
        unique_users = fields[i]
        print "Number of unique users:", fields[i]
    elif i == 31:
        if unique_users > 0:
            average = number_tweets / unique_users
        print("Average of tweets per user: {:.1f}".format(average))
    elif i in fields.keys():
        if i%2 == 0:
            print "Spam:", fields[i]
        else:
            print "Not Spam:", fields[i]
    else:
	pass

print fields
