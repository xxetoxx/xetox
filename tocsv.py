import pandas as pd

#jsonのprevを取り除くプログラム


df=pd.read_json("nikkei.json")
df.to_csv("nikkei.csv",index=False)


