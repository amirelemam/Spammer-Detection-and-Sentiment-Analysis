files = ["0-500k", "500k-1M", "1M-1.5M", "1.5M-2M", "2M-2-5M", "2.5M-3M",
         "3M-3-5M", "3.5M-4M", "4M-4-5M", "4.5M-5M","5M-5-5M", "5.5M-6M",
         "6M-6-5M", "6.5M-7M", "7M-7-5M"]
for item in files:
    with open('paris_' + item + '.txt', 'w') as _f:
        pass

count = 0
i = 0
with open("paris.txt", 'r') as f:
    f.seek(0)
    for row in f:
        count += 1
        print (count, files[i])
        with open('paris_' + files[i] + '.txt', 'a+') as f1:
            f1.write(row)
        if count == 500000:
            count = 0
            i += 1

for item in files:
    c_row = 0
    with open('paris_' + item + '.txt', 'r') as f1:
        for row in f1:
            c_row += 1
    print c_row
