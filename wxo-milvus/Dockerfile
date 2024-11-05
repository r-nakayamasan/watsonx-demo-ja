# ベースイメージ
FROM python:3.9

# 作業ディレクトリを設定
WORKDIR /app

# アプリケーションファイルと依存関係をコピー
COPY app.py /app
COPY requirements.txt /app

# 依存関係をインストール
RUN pip install -r requirements.txt

# コンテナを起動した際に実行されるコマンド
CMD ["python", "app.py"]
