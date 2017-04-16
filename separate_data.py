count = 0
with open('paris.txt', 'r') as f:
    for line in f:
        count += 1
        if count <= 500000:
            with open('paris_0-500k.txt', 'a') as _f:
                _f.write(line)
        elif count > 500000 and count <= 1000000:
            with open('paris_500k-1M.txt', 'a') as _f:
                _f.write(line)
        elif count > 1000000 and count <= 1500000:
            with open('paris_1M-1.5M.txt', 'a') as _f:
                _f.write(line)
        elif count > 1500000 and count <= 2000000:
            with open('paris_1.5M-2M.txt', 'a') as _f:
                _f.write(line)
        elif count > 2000000 and count <= 2500000:
            with open('paris_2M-2-5M.txt', 'a') as _f:
                _f.write(line)
        elif count > 2500000 and count <= 3000000:
            with open('paris_2.5M-3M.txt', 'a') as _f:
                _f.write(line)
        elif count > 3000000 and count <= 3500000:
            with open('paris_3M-3-5M.txt', 'a') as _f:
                _f.write(line)
        elif count > 3500000 and count <= 4000000:
            with open('paris_3.5M-4M.txt', 'a') as _f:
                _f.write(line)
        elif count > 4000000 and count <= 4500000:
            with open('paris_4M-4-5M.txt', 'a') as _f:
                _f.write(line)
        elif count > 4500000 and count <= 5000000:
            with open('paris_4.5M-5M.txt', 'a') as _f:
                _f.write(line)
        elif count > 5000000 and count <= 5500000:
            with open('paris_5M-5-5M.txt', 'a') as _f:
                _f.write(line)
        elif count > 5500000 and count <= 6000000:
            with open('paris_5.5M-6M.txt', 'a') as _f:
                _f.write(line)
        elif count > 6000000 and count <= 6500000:
            with open('paris_6M-6-5M.txt', 'a') as _f:
                _f.write(line)
        elif count > 6500000 and count <= 7000000:
            with open('paris_6.5M-7M.txt', 'a') as _f:
                _f.write(line)
        elif count > 7000000 and count <= 7500000:
            with open('paris_7M-7-5M.txt', 'a') as _f:
                _f.write(line)
        elif count > 7500000 and count <= 8000000:
            with open('paris_7.5M-8M.txt', 'a') as _f:
                _f.write(line)

