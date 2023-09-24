FROM python:3.11.4

# 必要な依存関係をインストール
RUN apt-get update && apt-get install -y \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install ibm-generative-ai==0.2.5

WORKDIR /app
RUN chmod 777 /app

COPY app.py .

ENV SENTENCE_TRANSFORMERS_HOME /app
# cache ディレクトリを作成
RUN mkdir -p /app/cache
RUN chmod 777 /app/cache
# TRANSFORMERS_CACHE の場所を設定
ENV TRANSFORMERS_CACHE /app/cache

RUN mkdir -p /app/matplotlib_cache
RUN chmod 777 /app/matplotlib_cache
ENV MPLCONFIGDIR /app/matplotlib_cache

ENV FONTCONFIG_PATH=/etc/fonts

# Gradioが使用するポートを公開
EXPOSE 8000

CMD [ "python", "app.py" ]
