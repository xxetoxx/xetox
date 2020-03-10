#readlines()
#入力ストリームから全ての行を読み込み、行のリストとして返します。
#difference(other, …)
#set に含まれて、かつ、全ての other に含まれない要素を持つ、新しい集合を返します。

def make_lines_set(txt):
    f = open(txt,encoding="utf-8")
    lines = f.readlines()
    sets = set(lines)
    f.close()
    return sets


bigs = make_lines_set('big.txt') #bigsにのみ存在する行を抽出
smalls = make_lines_set('small.txt')

results = bigs.difference(smalls)
for result in results:
    print(result)
