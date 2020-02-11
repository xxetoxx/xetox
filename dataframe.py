import pandas as pd
import csv
import os
import codecs
import numpy as np

#to_csv引数のinplaceをTrueにすればUnnmaedカラムが追加されずに済む。
#pd.to_csv('output.csv',inplace=True)

 #data_namesort save
'''
with codecs.open("./data/train_data.csv", "r", "Shift-JIS", "ignore") as file:
    df = pd.read_csv(file, delimiter=",", names=["年","月","日","馬名","馬番","枠番","年齢","性別","馬体重","斤量","場所","頭数","距離","馬場状態","天候","人気","単勝オッズ","確定着順","タイムS","着差タイム","トラックコード"])
    print(df)

df_sort_name=df.sort_values(['馬名',"年","月","日"], ascending=[True, False, False, False])

print(df_sort_name)

df_sort_name=df_sort_name.reset_index(drop=True)
df_sort_name.to_csv("./data/train_data_namesort.csv")


'''

datasize=50
element=21

DF=pd.DataFrame()


df = pd.read_csv("./data/train_data_namesort_test.csv", delimiter=",", index_col=0) #nunamed;0  非表示　
#print(df)
dfname=df['馬名']
#print(dfname.iloc[:,1])
for i in range(datasize-5):
    if dfname.iloc[i] == dfname.iloc[i+1]:
        dfcon1=pd.concat([df.iloc[i],df.iloc[i+1]],axis=0)  #df; all data, dfname; 'bamei' column only
        if all(dfcon1['馬名'] == dfname.iloc[i+2]):    #https://note.nkmk.me/python-pandas-multiple-conditions/
            dfcon2=pd.concat([dfcon1,df.iloc[i+2]],axis=0)
            #print(dfcon2)
            if all(dfcon2['馬名'] == dfname.iloc[i+3]):
                dfcon3=pd.concat([dfcon2,df.iloc[i+3]],axis=0)
                #print(dfcon2)
                if all(dfcon3['馬名'] == dfname.iloc[i+4]):
                    dfcon4=pd.concat([dfcon3,df.iloc[i+4]],axis=0)
                    #print(dfcon4)
                    if all(dfcon4['馬名'] == dfname.iloc[i+5]):
                        dfcon5=pd.concat([dfcon4,df.iloc[i+5]],axis=0)
                        #(dfcon5.shape) #datasize=40 -> thenumberof dfcon5 = 32 ok

                        dfcon5.index=["年","月","日","馬名","馬番","枠番","年齢","性別","馬体重","斤量","場所","頭数","距離","馬場状態","天候","人気","単勝オッズ","確定着順","タイムS","着差タイム","トラックコード" ,\
                        "年1","月1","日1","馬名1","馬番1","枠番1","年齢1","性別1","馬体重1","斤量1","場所1","頭数1","距離1","馬場状態1","天候1","人気1","単勝オッズ1","確定着順1","タイムS1","着差タイム1","トラックコード1", \
                        "年2","月2","日2","馬名2","馬番2","枠番2","年齢2","性別2","馬体重2","斤量2","場所2","頭数2","距離2","馬場状態2","天候2","人気2","単勝オッズ2","確定着順2","タイムS2","着差タイム2","トラックコード2",\
                        "年3","月3","日3","馬名3","馬番3","枠番3","年齢3","性別3","馬体重3","斤量3","場所3","頭数3","距離3","馬場状態3","天候3","人気3","単勝オッズ3","確定着順3","タイムS3","着差タイム3","トラックコード3", \
                        "年4","月4","日4","馬名4","馬番4","枠番4","年齢4","性別4","馬体重4","斤量4","場所4","頭数4","距離4","馬場状態4","天候4","人気4","単勝オッズ4","確定着順4","タイムS4","着差タイム4","トラックコード4", \
                        "年5","月5","日5","馬名5","馬番5","枠番5","年齢5","性別5","馬体重5","斤量5","場所5","頭数5","距離5","馬場状態5","天候5","人気5","単勝オッズ5","確定着順5","タイムS5","着差タイム5","トラックコード5"]

                        DF=pd.concat([DF,dfcon5],axis=1,sort=False)


#l.reshape(int (len(dfcon5)/(element*6)),int (element*6))
DF=DF.T
print(DF)
