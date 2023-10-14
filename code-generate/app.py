print("インポート開始")
from typing import Iterator

import os

import gradio as gr
import torch

from threading import Thread

from genai.model import Credentials, Model #まず先にgenaiをimport、次にibm-generative-aiをインポート

from genai.schemas import GenerateParams

from genai.schemas import TokenParams

from queue import Queue

# 変数

DEFAULT_SYSTEM_PROMPT = """以下は、さまざまな人々とAIのテクニカル・アシスタントとの一連の対話である。アシスタントは、親切で、丁寧で、正直で、洗練されていて、感情的で、謙虚だが知識豊富であろうとする。このアシスタントは、コードの質問を喜んで手伝い、何が必要かを正確に理解するために最善を尽くします。また、偽の情報や誤解を招くような情報を与えないようにし、正しい答えが完全にわからないときには注意を促します。とはいえ、このアシスタントは実用的で、本当にベストを尽くしてくれる。

------

Japanese：<指示文>
Assistant：```<回答>```"""

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4000

# 使用するモデル
llm_id = 'bigcode/starcoder'

#LLMの準備
# APIのkeyを挿入
# api_key = input("あなたのBAM APIを入力して下さい")
api_key = os.environ.get("API_KEY")
if not api_key:
    raise ValueError("API_KEY environment variable is not set!")

api_endpoint = "https://bam-api.res.ibm.com/v1/"
creds = Credentials(api_key=api_key, api_endpoint=api_endpoint)

# # トークナイザー
tokenizer = Model(llm_id, params=TokenParams, credentials=creds)

#テキストボックスのクリアー
def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return '', message

#メッセージインプット
def display_input(message: str,
                  history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ''))
    return history

def delete_prev_fn(
        history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''

def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:

#     texts = [f'''<s>[INST] <<SYS>>
# {system_prompt}
# <</SYS>>

# あなたは「watsonx.ai」という名前のチャットボットです。ユーザーと明るく会話します。日本語で会話を続けてください。
# ''']

    texts = [f"""{system_prompt}

-----"""]
    
    # The first user input is _not_ stripped(strip=前後の空白が削除される)
    do_strip = False
    
    # チャット履歴をtextesに追加
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
#         texts.append(f'ユーザー: {user_input} [/INST] あなた: {response.strip()} </s><s>[INST] ')
        texts.append(f"""

Japanese：{user_input}
Assistant：{response.strip()}""")
        
    #2回目以降
    #messageは今回のメッセージ
    if message is not None:
        message = message.strip() if do_strip else message
    else:
        message = ""

#     texts.append(f"{message} [/INST] あなた: ")
    texts.append(f"""

Japanese：{message} 
Assistant：
""")

    return ''.join(texts)


def get_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    prompt = get_prompt(message, chat_history, system_prompt)
    tokenized_response = tokenizer.tokenize([prompt], return_tokens=True)
    input_ids = tokenized_response[0].token_count
    return input_ids

def run(message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int) -> Iterator[str]:
    
    prompt = get_prompt(message, chat_history, system_prompt)
    q = Queue()

    params = GenerateParams(
        decoding_method="sample",
        stream=True,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=1.0,
#         random_seed=1,
#         beam_width=1,
        stop_sequences=["-----"]
    )
    
    llm = Model(model=llm_id, params=params, credentials=creds)
    
    print("テキストを出力します")
    print(params)
    print(prompt)
    
    def threaded_generate(q: Queue, prompt: str):
        for chunk in llm.generate_stream([prompt]):
            q.put(chunk)
        q.put(None)
    
    t = Thread(target=threaded_generate, args=(q, prompt))
    t.start()

    outputs = []
    
    while True:
        chunk = q.get()
        if chunk is None:
            break
        if chunk.generated_text is not None:
            outputs.append(chunk.generated_text)
            print(outputs)
        yield ''.join(outputs)
        

def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError
    
    #入力受け取り
    history = history_with_input[:-1]
    
    #出力
    generator = run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, '')]
    for response in generator:
        yield history + [(message, response)]
        

def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    generator = generate(message, [], DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
    for x in generator:
        pass
    return '', x

def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
    input_token_length = get_input_token_length(message, chat_history, system_prompt)
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        raise gr.Error(f'The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.')
    
with gr.Blocks(css='style.css') as demo:

    with gr.Group():
        chatbot = gr.Chatbot(label='Chatbot')
        with gr.Row():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder='メッセージを入力してね！',
                scale=10,
            )
            submit_button = gr.Button('送信✈️',
                                      variant='primary',
                                      scale=1,
                                      min_width=0)
    with gr.Row():
        retry_button = gr.Button('🔄  もう一度聞いてみる', variant='secondary')
        undo_button = gr.Button('↩️ 元に戻す', variant='secondary')
        clear_button = gr.Button('🗑️  会話を削除する', variant='secondary')

    saved_input = gr.State()

    with gr.Accordion(label='詳細設定', open=False):
        system_prompt = gr.Textbox(label='System prompt',
                                   value=DEFAULT_SYSTEM_PROMPT,
                                   lines=6)
        max_new_tokens = gr.Slider(
            label='Max new tokens',
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        )
        temperature = gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.3,
        )
        top_p = gr.Slider(
            label='Top-p (nucleus sampling)',
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95,
        )
        top_k = gr.Slider(
            label='Top-k',
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        )


    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    button_event_preprocess = submit_button.click(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )

    clear_button.click(
        fn=lambda: ([], ''),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )

print("処理完了")

demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=8000)
