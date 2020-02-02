# XETOX

**TODO:Data loader,Preprocess data,**

## Description

This is a PyTorch implementation of horse racing prediction

## Installation

[Install Anaconda](https://www.anaconda.com/)

[Install PyTorch](https://pytorch.org/)

```
git clone https://github.com/ukyoukatsura/xetox.git
```

## Usage

```
cd xetox
Python xetox.py
```

## Author

Katsura Ukyo

## Data Pre-Processing

* 馬番

* 枠番

* 年齢

* 性別
    * 牡:1
    * 牝:2

* 馬体重

* 斤量

* 場所
    * 札幌:1
    * 函館:2
    * 福島:3
    * 新潟:4
    * 東京:5
    * 中山:6
    * 中京:7
    * 京都:8
    * 阪神:9
    * 小倉:10

* 頭数

* 距離

* 馬場状態
    * 良:1
    * 稍:2
    * 重:3
    * 不:4

* 天候
    * 晴:1
    * 曇:2
    * 雨:3
    * 小雨:4
    * 小雪:5
    * 雪:6

* 人気

* 単勝オッズ

* 確定着順

* タイムS
    * datetime:秒数

* 着差タイム

* トラックコード