import urllib

#data = urllib.request.urlopen("https://services.swpc.noaa.gov/text/discussion.txt")
#data = data.split('\n') # then split it into lines

for line in urllib.request.urlopen("https://services.swpc.noaa.gov/text/discussion.txt"):
    print(line.decode('utf-8'))