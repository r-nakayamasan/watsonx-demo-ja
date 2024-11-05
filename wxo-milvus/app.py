from flask import Flask, request, jsonify
import requests
import json
from dotenv import load_dotenv
import os
import logging

load_dotenv()  # .envファイルをロード
service_b_api_key = os.getenv("SERVICE_B_API_KEY")
service_b_url = os.getenv("SERVICE_B_API_URL")
huggingface_url = os.getenv("huggingface_url")
huggingface_api = os.getenv("huggingface_api")

app = Flask(__name__)

@app.route('/proxy', methods=['POST'])
def proxy():
    # サービスAからのリクエストを取得し、スキーマを変換
    data_from_service_a = request.json
    transformed_data_for_b = transform_for_service_b(data_from_service_a)

    url = service_b_url+"/api/v1/search"
    # サービスBにリクエストを送信
    response = requests.post(url=url, json=transformed_data_for_b, auth=("apikey", service_b_api_key))
    data_from_service_b = response.json()
    print(data_from_service_b)

    # サービスBからのレスポンスをサービスA用のスキーマに変換
    transformed_data_for_a = transform_for_service_a(data_from_service_b)
    return jsonify(transformed_data_for_a)


def transform_for_service_b(data):
    # サービスAのリクエストデータを読み込み
    query = data.get("query", "")
    filter_criteria = data.get("filter", "")
    metadata = data.get("metadata", {})

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        "Authorization": f"Bearer {huggingface_api}"
    }
    data = {
        "inputs": ["query"]
    }

    response = requests.post(url=huggingface_url, headers=headers, json=data)
    data = response.json()
    query = data[0]
    print(query)

    # サービスBのリクエストデータを構築
    transformed_data = {
        "vectors": [query],  # ここに適切なベクトルデータを入力
        "dsl_type": 1,
        "output_fields": ["text"],
        "search_params": [
            {"key": "anns_field", "value": "vector"},
            {"key": "topk", "value": "1"},
            {"key": "params", "value": "{\"nprobe\": 10}"},
            {"key": "metric_type", "value": "L2"},
            {"key": "round_decimal", "value": "-1"}
        ],
        "collection_name": "MLB"
    }

    # 必要に応じて、クエリやフィルタをベクトル形式や追加パラメータに変換
    # 例: queryの内容に基づいて`vectors`フィールドを設定（内容によるカスタム実装が必要）

    return transformed_data

def transform_for_service_a(data):
    # サービスBのレスポンスから`text`データを抽出
    text_data = data.get("results", {}).get("fields_data", [])[0] \
                .get("Field", {}).get("Scalars", {}).get("Data", {}).get("StringData", {}).get("data", [""])[0]

    # サービスAのフォーマットに変換
    transformed_data = {
        "search_results": [
            {
                "response_metadata": {
                    "title": "検索結果",
                    "body": text_data
                }
            }
        ]
    }
    return transformed_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
