import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import difflib


# d = difflib.Differ()
# diff = d.compare(text1_lines, text2_lines)
# print('\n'.join(diff))

time_flag = True
#永久に実行させます
#while True:
url = "https://www.ai-yuma.com/"

r = requests.get(url)

soup = BeautifulSoup(r.text, "html.parser")
#elems=soup.select('.entry-content')
elememp=[]
#次のページへ

nextpages=soup.select(".pager-next")
for i in range(50):
    nextpages = soup.select(".pager-next")
    elems = soup.select('.entry-content')
    elememp.append(elems)
    for link in nextpages:
        nextpagelink=link.a.get('href') #aなしではlinkでない
    next=requests.get(nextpagelink)
    soup = BeautifulSoup(next.text, "html.parser")



# print(soup.prettify()) # HTMLをインデントすることができます
#print(soup.title.text)  # titleのテキストのみ出力
#print(elems)  # titleのテキストのみ出力

# for elem in elems:
#     print(elem)
print(elememp)
# 現在の時刻を年、月、日、時、分、秒で取得します
time_ = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
print(time_)
# 1分間待機します
time.sleep(60)
    #continue
