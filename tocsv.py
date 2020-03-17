import pandas as pd
import re

#with open("yuma.txt","w",encoding="utf8") as f:
for line in open("aaa.txt","r",encoding="utf8"):
     #f.write(re.sub(r'.+中央の結果|.+([0-9],[0-9],[0-9]).+',"",line)) #中央の結果記事を削除する
     #print(re.search(r'[0-9]{8}',line)) #日付
     #print(re.search(r'\d+R',line)) #レース番号
     #print(re.search(r'((?<=\d{8}\s).*(?=\d{2}R)|(?<=\d{8}\s).*(?=\d{1}R))',line)) #レース場
     #print(re.findall(r'(\d+)【',line)) #馬番
     #print(re.findall(r'(\d+\.\d+)%】',line)) #勝率


# with open("aaa.txt","r",encoding="utf8") as f:
#     text=f.read()
# print(text)


'''
#delete_\n
with open("yuma_resultdeletev2.txt","w",encoding="utf8") as f:
     for line in open("yuma_resultdelete.txt","r",encoding="utf8"):
         if line[0] != "\n":
               f.write(line)
'''
