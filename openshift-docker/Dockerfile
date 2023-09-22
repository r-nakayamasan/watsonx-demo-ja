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

COPY app.py .

CMD [ "python", "app.py" ]