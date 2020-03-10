import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import difflib
import csv
import re
'''
csv化コード
with open("a.txt","r",encoding="utf-8") as f:
    text=f.read()

text_mod=re.sub('◯|▲|△|☆|－',",",text)
with open("a.csv","w",encoding="utf-8") as f:
    f.write(text_mod)

'''

# d = difflib.Differ()
# diff = d.compare(text1_lines, text2_lines)
# print('\n'.join(diff))

time_flag = True
#永久に実行させます
#while True:
url = "https://www.ai-yuma.com/"

r = requests.get(url)

soup = BeautifulSoup(r.text, "html.parser")
#elems=soup.select('.entry-content') #予想記事のclass
elems=[]


#nextpages=soup.select(".pager-next") #次のページへのclass
for i in range(3): #何回次のページへ飛ぶか(1だと飛ばない)
    nextpages = soup.select(".pager-next")
    tmpelem = soup.select('.entry-content')
    elems.append(tmpelem)
    for link in nextpages:
        nextpagelink=link.a.get('href') #aなしではlinkが取得できない
    next=requests.get(nextpagelink) #現在ページの次のページ取得
    soup = BeautifulSoup(next.text, "html.parser")

#p = re.compile(r"<[^>]*?>")
#p.sub("", )


# with open('yuma.csv', 'w') as file:
#     writer = csv.writer(file, lineterminator='\n')
#     writer.writerows(elems)

#file書き込み
with open("yuma1.txt","w") as f:
    for elem in elems:
        for ele in elem:
            f.write(ele.text)


# print(soup.prettify()) # HTMLをインデントできる
#print(soup.title.text)  # titleのテキストのみ出力
#print(elems)

'''
#予想結果の配列を二回for文で要素を取ってきて文字列へ、そしてtext部分だけをprintしたが、elem.textには１レース分しか入らない
for elem in elems:
    for elem in elem:
        print(elem.text)
        
'''

# 現在の時刻を年、月、日、時、分、秒で取得します
time_ = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
print(time_)
# 1分間待機します
#time.sleep(60)
    #continue
