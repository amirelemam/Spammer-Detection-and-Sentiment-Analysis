import json

files = ["0-500k", "500k-1M", "1M-1.5M", "1.5M-2M", "2M-2-5M", "2.5M-3M",
         "3M-3-5M", "3.5M-4M", "4M-4-5M", "4.5M-5M","5M-5-5M", "5.5M-6M",
         "6M-6-5M", "6.5M-7M", "7M-7-5M"]

with open("trained_data.txt", 'r') as t:
    termos = t.read()

for i in range(len(files)):
    print("Processando arquivo {}".format(files[i]))
    with open('paris_' + files[i] + '.txt',"r") as f, open('paris__' + files[i] + '.txt',"w") as g:
        for line in f:
            tweet = json.loads(line)
            if tweet['timestamp_ms'] not in termos:
                g.write(line)
