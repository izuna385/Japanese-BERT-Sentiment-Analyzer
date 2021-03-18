# Japanese-BERT-Sentiment-Analyzer
* twitterデータセットを用いたポジ/ネガ/中性の判定モデル

# 手順
## データセットのダウンロード
* [鈴木研ホームページ](http://www.db.info.gifu-u.ac.jp/data/Data_5d832973308d57446583ed9f) よりデータセットを利用させて頂いています。

* `./dataset/` を詳しくは御覧ください。

## 実験環境
```python
$ conda create -n allennlp python=3.7
$ conda activate allennlp
$ pip install -r requirements.txt
```

## 実験
```
$ python3 train.py
```

ここまでで、訓練されたモデルのパラメータが`./serialization_dir/` 下に保存される

## APIのコンテナ化
```
$ docker build -t jsa:latest .
$ docker run -d -itd -p 8000:8000 jsa
```
（コンテナ化しない場合）
```
$ uvicorn app:app --reload --port 8000 --host 0.0.0.0 --log-level trace
```

## 使用法
```
$ curl -X 'POST' 'http://localhost:8000/sentiment/' -H 'accept: application/json' \
       -H 'Content-Type: application/json' \
       -d '{
            "sentence": "今日はいい天気"
           }'

>> {"probs":
        {
         "neutral":0.8089876174926758,
         "negative":0.015650086104869843,
         "positive":0.17536230385303497
         }
    }
```
