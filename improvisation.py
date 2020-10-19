"""
import codecs
import os
import sys
import time
import traceback
import win32con
import win32evtlog
import win32evtlogutil
import winerror


# ----------------------------------------------------------------------
def getAllEvents(server, logtypes, basePath):
    if not server:
        serverName = "localhost"
    else:
        serverName = server
    for logtype in logtypes:
        path = os.path.join(basePath, "%s_%s_log.log" % (serverName, logtype))
        getEventLogs(server, logtype, path)


# ----------------------------------------------------------------------
def getEventLogs(server, logtype, logPath):

    print
    ("Logging %s events" % logtype)
    log = codecs.open(logPath, encoding='utf-8', mode='w')
    line_break = '-' * 80

    log.write("\n%s Log of %s Events\n" % (server, logtype))
    log.write("Created: %s\n\n" % time.ctime())
    log.write("\n" + line_break + "\n")
    hand = win32evtlog.OpenEventLog(server, logtype)
    total = win32evtlog.GetNumberOfEventLogRecords(hand)
    print
    ("Total events in %s = %s" % (logtype, total))
    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
    events = win32evtlog.ReadEventLog(hand, flags, 0)
    evt_dict = {win32con.EVENTLOG_AUDIT_FAILURE: 'EVENTLOG_AUDIT_FAILURE',
                win32con.EVENTLOG_AUDIT_SUCCESS: 'EVENTLOG_AUDIT_SUCCESS',
                win32con.EVENTLOG_INFORMATION_TYPE: 'EVENTLOG_INFORMATION_TYPE',
                win32con.EVENTLOG_WARNING_TYPE: 'EVENTLOG_WARNING_TYPE',
                win32con.EVENTLOG_ERROR_TYPE: 'EVENTLOG_ERROR_TYPE'}

    try:
        events = 1
        while events:
            events = win32evtlog.ReadEventLog(hand, flags, 0)

            for ev_obj in events:
                the_time = ev_obj.TimeGenerated.Format()  # '12/23/99 15:54:09'
                evt_id = str(winerror.HRESULT_CODE(ev_obj.EventID))
                computer = str(ev_obj.ComputerName)
                cat = ev_obj.EventCategory
                ##        seconds=date2sec(the_time)
                record = ev_obj.RecordNumber
                msg = win32evtlogutil.SafeFormatMessage(ev_obj, logtype)

                source = str(ev_obj.SourceName)
                if not ev_obj.EventType in evt_dict.keys():
                    evt_type = "unknown"
                else:
                    evt_type = str(evt_dict[ev_obj.EventType])
                log.write("Event Date/Time: %s\n" % the_time)
                log.write("Event ID / Type: %s / %s\n" % (evt_id, evt_type))
                log.write("Record #%s\n" % record)
                log.write("Source: %s\n\n" % source)
                log.write(msg)
                log.write("\n\n")
                log.write(line_break)
                log.write("\n\n")
    except:
        print
        (traceback.print_exc(sys.exc_info()))

    print
    ("Log creation finished. Location of log is %s" % logPath)




import matplotlib.pyplot as ASS
import math,statistics,random
import numpy as np


def main():
    In = open('input.txt', 'r')
    out = open('output.txt', 'w')
    n = int(In.read())
    res = 0
    for i in range(n- 1, 1, -1):
        if (n % i == 0):
            res += i
    if res % 2 == 0:
        out.write('YES\n')
    else:
        out.write('NO\n')

if __name__=="__main__":
    main()


import asyncio

async def main():
    print('Hello ...')

    print('... World!')

# Python 3.7+
asyncio.run(main())

import re, os, sys, argparse, time
import ffmpeg, youtube_dl
import urllib.request
import urllib.parse

urlopen = urllib.request.urlopen
encode = urllib.parse.urlencode
retrieve = urllib.request.urlretrieve
cleanup = urllib.request.urlcleanup()

# function to retrieve video title from provided link
def video_title(url):
    try:
        webpage = urlopen(url).read()
        title = str(webpage).split('<title>')[1].split('</title>')[0]
    except:
        title = 'Youtube Song'

    return title


# download from a list of songs or links
def list_download(song_list=None):
    if not song_list:
        song_list = ""  # get the file name to be opened
    # find the file and set fhand as handler
    try:
        fhand = open(song_list, 'r')
    except IOError:
        print('File does not exist')
        exit(1)

    # Iterating over the lines in file
    for song in fhand:
        single_download(song)

    fhand.close()


# download directly with a song name or link
def single_download(song=None):
    if not (song):
        song = "https://www.youtube.com/watch?v=O4Mv7QTwuZI"

    if "youtube.com/" not in song:
        # try to get the search result and exit upon error
        try:
            query_string = encode({"search_query": song})
            html_content = urlopen("http://www.youtube.com/results?" + query_string)
            search_results = re.findall(r'href=\"\/watch\?v=(.{11})', html_content.read().decode())
        except:
            print('Network Error')
            return None
        command = 'youtube-dl --embed-thumbnail --no-warnings --extract-audio --audio-format mp3 -o "%(title)s.%(ext)s" ' + \
                  search_results[0]
    else:
        command = 'youtube-dl --embed-thumbnail --no-warnings --extract-audio --audio-format mp3 -o "%(title)s.%(ext)s" ' + song[song.find("=") + 1:]
    os.system(command)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--single", help="for single song download.")
    ap.add_argument("-l", "--list", help="for list of song download")
    #list_download()
    single_download()

if __name__ == '__main__':
    main()  # run the main program



import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

im = Image.open("photo.png")
text = pytesseract.image_to_string(im, lang='ukr')

print(text)

"""

import base64
# Creating a string
s = ""# Encoding the string into bytes
b = s.encode("UTF-8")
# Base64 Encode the bytes
e = base64.b64encode(b)
# Decoding the Base64 bytes to string
s1 = e.decode("UTF-8")
# Printing Base64 encoded string
print("Base64 Encoded:", s1)
# Encoding the Base64 encoded string into bytes
d = base64.b64decode("0JTQviDRgNC10YfRliwg0Y8g0L/QsNC8J9GP0YLQsNGOINGP0Log0LTQsNCy0LDQsiDQvtCx0ZbRhtGP0L3QutGDINCy0LHQuNGC0Lgg0YLQvtCz0L4g0YXRgtC+INCx0YPQtNC1INGC0LXQsdC1INCx0ZbRgdC40YLQuC4g0JDQu9C1INGJ0L4g0Y/QutGJ0L4g0YbQtSDQsdGD0LTQtSDRjz8g0K/QutGJ0L4g0YbQtSDQsdGD0LTRgyDRjywg0LTQsNGOINC/0LjRgdGM0LzQvtCy0YMg0YDQvtC30L/QuNGB0LrRgyDQv9GA0L4g0YLQtSwg0YnQviDQstC4INC30LzQvtC20LXRgtC1INC30LDRgdC90Y/RgtC4INGC0LUsINGP0Log0Y8g0YHQtdCx0LUg0LLQsSfRji4g0JzQvtC20YMg0L3QsNCy0ZbRgtGMINCy0ZbQtNC00LDRgtC4INCy0LDQvCDQstGB0LUg0L/QvtGC0YDRltCx0L3QtSDQtNC70Y8g0YbRjNC+0LPQviwg0LAg0YLQsNC8INCy0YHQtSDQt9Cw0YXQvtGH0LXRgtC1")
s2 = d.decode("UTF-8")
print(s2)

