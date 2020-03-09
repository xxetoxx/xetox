import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from config import config, update_recent_article

time_flag = True



def extract_pick_up(soup=soup):
    """
    指定されたページのhtmlを読み込み、最新記事を抜粋してくる
    """
    columns = soup.find_all("article", class_="entry-content")
    return columns[0:13]

"""
def extract_url(column):

    columns @params: コラム

    コラムのurlを取得する

    column_html = BeautifulSoup(str(column), "lxml")
    url = column_html.find("a").get("href")
    return url
"""

def extract_title(column):
    """
    columns @params: コラム

    コラムのタイトルを取得する
    """
    column_html = BeautifulSoup(str(column), "lxml")
    title = column_html.find("h2").string
    return title

def extract_update_article(columns):
    """
    更新された記事を抽出
    更新前の最新の記事のindexはconfig.iniで管理
    """
    # 最新の記事番号を取得
    recent_article = config['web_info']['recent_article']
    # 更新された記事のurlを取得
    article_list = []
    update_start = False
    # 逆順に取得していき、最新記事の次からの記事のurlを取得
    for column in reversed(columns):
        url = extract_url(column)
        # 記事番号取得
        article_num = url.replace(config['web_info']['url']+"/", "")

        # タイトル取得
        title = extract_title(column)
        # 取得すべきurl
        if update_start:
            article_list.append([title, url])
            recent_article = article_num
            continue
        # 前回取得した最新の記事かどうか判定
        if recent_article == article_num:
            update_start = True

    # config更新
    update_recent_article(recent_article)
    if article_list == []:
        return [["更新記事はありません", "https://kintore.site"]]
    return article_list


if __name__ == "__main__":
    columns = extract_pick_up()
    #url = extract_url(columns[0])
    article_list = extract_update_article(columns)
    print(article_list)
    # 永久に実行させます
    # while True:

    url = "https://www.ai-yuma.com/"

    r = requests.get(url)

    soup = BeautifulSoup(r.text, "html.parser")

    # print(soup.prettify()) # HTMLをインデントすることができます
    print(soup.title.text)  # titleのテキストのみ出力
    # 現在の時刻を年、月、日、時、分、秒で取得します
    time_ = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(time_)
    # 1分間待機します
    # time.sleep(60)
    # continue