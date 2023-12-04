from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from llama_index import set_global_tokenizer
from transformers import AutoTokenizer

# use Huggingface embeddings
from llama_index.embeddings import HuggingFaceEmbedding


model_path = './llama-2-7b-chat.Q4_0.gguf'
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=None,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

set_global_tokenizer(AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


app = Flask(__name__)
app.static_folder = 'template/src'

socketio = SocketIO(app)

# create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

documents = SimpleDirectoryReader('Wiki/data').load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()


@app.route('/')
def index():
    return render_template("index.html")


@socketio.on('connect')
def on_connect():
    room_id = request.sid  # Получаем уникальный идентификатор сессии пользователя
    join_room(room_id)  # Добавляем пользователя в комнату с его идентификатором сессии


@socketio.on('query')
def handle_query(data):
    query = data.get('query')  # Получаем значение поля 'query' из словаря data
    room_id = request.sid  # Получаем уникальный идентификатор сессии пользователя

    response = query_engine.query(query)
    response_text = str(response)

    # Отправляем ответ только в комнату с уникальным идентификатором сессии пользователя
    socketio.emit('response', response_text, room=room_id)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5111)
