import pandas as pd
import re


#\nなくす
#with open("aaa2.txt", "w", encoding="utf8") as f:

"""
line
20200315 高知10R ファイナルレース　Ｃ２　－６記者選抜
◎ 6【23.0%】 テレグライド(倉兼育)◯ 1【14.2%】 デュパルクカズマ(岡村卓)▲ 4【10.2%】 プリサイスホウプ(佐原秀)△ 7【10.0%】 ドラゴンチャンプ(★濱尚美)☆ 9【 8.7%】 マロンスライム(西川敏)－ 3【 6.9%】 トウカイバレット(西森将)－10【 6.3%】 メメント(☆塚本雄)－12【 5.9%】 コパノハミルトン(大澤誠)－ 2【 5.4%】 マエムキ(木村直)－11【 4.9%】 フユハナビ(林謙佑)－ 8【 2.8%】 クラフツマンシップ(妹尾浩)－ 5【 1.1%】 ハニーフェイバー(☆多田誠)
"""

for line in open("aaa.txt","r",encoding="utf8"):
    if line[0] != "\n":
        #ここから考える
       # print(re.sub(r'.+中央の結果|.+([0-9],[0-9],[0-9]).+',"",line)) #中央の結果記事を削除する
       # print(re.search(r'[0-9]{8}',line)) #日付
        #print(re.search(r'[0-9]{,2}R',line)) #レース番号
        print(re.search(r'.*(?=\d+R)',line)) #レース場()\d{1}R|\d{2}R でやってみる



# with open("aaa.txt","r",encoding="utf8") as f:
#     text=f.read()
# print(text)



