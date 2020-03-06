import urllib3
import requests
from bs4 import BeautifulSoup

url="https://www.ai-yuma.com/"

r =requests.get(url)

soup = BeautifulSoup(r.text, "html.parser")

#print(soup.prettify()) # HTMLをインデントすることができます
print(soup.title.string) #titleのテキストのみ出力
