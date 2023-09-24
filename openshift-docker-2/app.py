import os
import requests
import gradio as gr
# from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
from genai.extensions.langchain import LangChainInterface
from langchain.document_loaders import UnstructuredPDFLoader #pip install pdfminer.six
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.indexes import VectorstoreIndexCreator
from genai.model import Credentials, Model #まず先にgenaiをimport、次にibm-generative-aiをインポート
from genai.schemas import GenerateParams
# from langchain.vectorstores.faiss import FAISS
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
# import torch
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain import PromptTemplate, LLMChain
#使用するPDF

# print(os.getcwd())
# folder_path = './docs'
# print(os.listdir(folder_path))

#とりあえずファイルで指定
folder_path = []
url = "https://rag-odf-folder.s3.jp-tok.cloud-object-storage.appdomain.cloud/docs/MLB_ja.txt"
response = requests.get(url)

#変更できる変数

# 読み込んだPDFのテキスト長から自動的にchunk_sizeを決定
# chunk_limit = 500
chunk_limit = 600

# chunk_sizeの10%をchunk_overlapとする
# chunk_overlap = 500
chunk_overlap = 200

#ベクターストアの場所
# vec = "article_" +  str(chunk_limit) + "_" + str(chunk_overlap)
# persist_directory="./RAG/vec_add/" + vec
#LLMの準備
# APIのkeyを挿入
# api_key = input("あなたのBAM APIを入力して下さい")
api_key = os.environ.get("API_KEY")
if not api_key:
    raise ValueError("API_KEY environment variable is not set!")

api_endpoint = "https://bam-api.res.ibm.com/v1/"

print("少々お待ちください")

print("ドキュメントを読み込んでいます")
creds = Credentials(api_key=api_key, api_endpoint=api_endpoint)
# PDFのローディング

new_documents = ""
new_documents = response.content.decode('utf-8')

# for fol in os.listdir(folder_path):
#     if os.path.splitext(fol)[1] == ".pdf":
#         # PDFのローディング
#         loader = UnstructuredPDFLoader(os.path.join(folder_path, fol)) #ローダー定義
#         document = loader.load()
#         new_documents += document[0].page_content
#         print(fol, "を読み込みました。")
#     elif os.path.splitext(fol)[1] == ".txt":
#         loader = TextLoader(os.path.join(folder_path, fol))
#         document = loader.load()
#         new_documents += document[0].page_content
#         print(fol, "を読み込みました。")
#     else:
#         print("読み込めませんでした")


print("==========================================")

print("今回のテキストは", len(new_documents),"文字です。")
# print(new_documents[0:100])
print("==========================================")
print("ベクトル化しています")

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# if vec in os.listdir("./RAG/vec_add/"):
#     #再利用
#     index = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# else:
    
#チャンクの分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_limit, chunk_overlap=chunk_overlap)
pdf_text = text_splitter.split_text(new_documents)

# ドキュメントの読みこみ
index = Chroma.from_texts(pdf_text, embeddings)
# Q&Aに使うLLM

params_qa = GenerateParams(
    decoding_method="greedy",
    min_new_tokens=3,
    max_new_tokens=1500,
    stream=False,
    repetition_penalty=1,
).dict() 

llm_qa = LangChainInterface(model='meta-llama/llama-2-13b-chat', credentials=creds, params=params_qa)
# リトリーバーの設定(ユーザーから質問を受け取り、ベクターストアに投げかけ、関連コードを返してくれるところ)
retriever = index.as_retriever()
# retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 50
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 4
retriever.search_kwargs['score_threshold'] = 0.9
qa_chain = RetrievalQA.from_chain_type(llm=llm_qa, 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    input_key="question")
def generate_qa(pdf_text, question):
    template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest Japanese assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

You should answer in Japanese. You shouldn't answer in any other than Japnanese.

{ground_text}

Question: {question}[/INST]Helpful answer in Japanese:"""
    prompt = PromptTemplate(
        template=template, 
        input_variables=["ground_text", "question"]
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm_qa, verbose=True)
    answer = llm_chain.predict(ground_text=pdf_text, question=question)
    # answer = ""
    return answer

latest_doc = ""
latest_q = None
latest_ans = None

print("完了しました")

# Chat UIの起動
def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    global latest_ans, latest_doc, latest_q  # グローバル変数を参照

    latest_q = history[-1][0]
    
    docs = ""

    docs = retriever.get_relevant_documents(query=latest_q)
    print(f"\n参考箇所: \n {docs}")
    latest_doc = ""
    for d in docs:
        latest_doc += d.page_content
        latest_doc += "\n"

    latest_ans = generate_qa(latest_doc, latest_q)
    print(f"\n回答: \n {latest_ans}")

    # docs = retriever.get_relevant_documents(query=latest_q)
    print(f"\n参考箇所: \n {docs}")
    
    history[-1][1] = latest_ans
    return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chat with PDF").style(height=600)
    
    txt = gr.Textbox(
        show_label=False,
        placeholder="Type your question and press enter",
    ).style(container=False)
    
    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    
    with gr.Row():
        with gr.Column(scale=0.6):
            chatbot
        


demo.launch(server_name="0.0.0.0", server_port=8000)