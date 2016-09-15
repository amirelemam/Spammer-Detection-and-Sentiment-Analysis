#!-*- conding: utf8 -*-
import simplejson as json
import codecs

count = 0
with codecs.open('json.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.replace("'", " ")
        try:
            line_object = json.loads(line.strip())
            print "%d %s" % (count, line_object['text'])
            count += 1
        except:
            pass
print count
