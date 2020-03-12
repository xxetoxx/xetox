import pandas as pd

#jsonのprevを取り除くプログラム


df=pd.read_json("nikkei.json")
df.to_csv("nikkei.csv",index=False)

”””
#リスト取り除く
"['09:01:00', 19709.05]"　[1]取得したらよさそう


”””
