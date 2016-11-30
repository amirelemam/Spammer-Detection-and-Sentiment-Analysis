import subprocess

files = ["0-500k", "500k-1M", "1M-1.5M", "1.5M-2M", "2M-2-5M", "2.5M-3M",
         "3M-3-5M", "3.5M-4M", "4M-4-5M", "4.5M-5M","5M-5-5M", "5.5M-6M",
         "6M-6-5M", "6.5M-7M", "7M-7-5M"]

for i in range(len(files)):
    subprocess.call(["python", "sentiment_analysis.py", files[i]])
