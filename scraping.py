import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import difflib
import csv
import re

#Ctrl + Alt + L 整形


url = "https://www.ai-yuma.com/"

r = requests.get(url)
pagecount=2
soup = BeautifulSoup(r.text, "html.parser")
# elems=soup.select('.entry-title')
# print(elems)
elems = []

# nextpages=soup.select(".pager-next") #次のページへのclass
for i in range(pagecount):  # 何回次のページへ飛ぶか(1だと飛ばない)
    nextpages = soup.select(".pager-next")  # link
    tmpelem = soup.select('.entry-content')  # 予想記事のclass
    title = soup.select('.entry-title')  # レース名
    elems.append(title)
    elems.append(tmpelem)
    for link in nextpages:
        nextpagelink = link.a.get('href')  # aなしではlinkが取得できない
    next = requests.get(nextpagelink)  # 現在ページの次のページ取得
    soup = BeautifulSoup(next.text, "html.parser")

#print(elems)
#print(str(elems[2][0].text) + str(elems[3][0].text))

with open("aaa.txt", "w", encoding="utf8") as f:
    for i in range(0,pagecount*2,2):
        for j in range(len(elems[i])):
            f.write(elems[i][j].text + elems[i+1][j].text)
