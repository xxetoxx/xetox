import pandas as pd
import codecs

# with codecs.open("./data/train_data.csv", "r", "Shift-JIS", "ignore") as file:
#     df = pd.read_csv(file, delimiter=",", names=["年","月","日","馬名","馬番","枠番","年齢","性別","馬体重","斤量","場所","頭数","距離","馬場状態","天候","人気","単勝オッズ","確定着順","タイムS","着差タイム","トラックコード"])
#     #print(df)

#df.to_csv("./data/train_data_utf_8.csv")
df = pd.read_csv("./data/train_data_utf_8_preprocessing.csv", delimiter=",")
#df = df.fillna(0.0)
print(df.isnull().values.sum() != 0)
#print(df)
#print(df[df["年"] == 20])
#print(df["天候"].value_counts())