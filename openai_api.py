# coding=utf-8
# Implements API for ChatGLM2-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.


import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModel
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import os
from dotenv import load_dotenv

load_dotenv()
tokens = os.environ.get("TOKENS").split("|")


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # 在这里执行验证逻辑，例如验证 token 的有效性
    if not is_valid_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def is_valid_token(token: str) -> bool:
    # 在这里实现验证 token 的逻辑
    # 返回 True 表示有效，返回 False 表示无效
    # 这里只是一个示例，你需要根据实际情况实现验证逻辑
    global tokens
    if token in tokens:
        return True
    return False
    # return token == "you token"


models = {}
tokenizers = {}
all_models = ["chatglm2-6b", "chatglm2-6b-32k"]


def load_model(model_name):
    global models, tokenizers
    if model_name in models:
        return models[model_name], tokenizers[model_name]
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(parent_dir, "hf.co_THUDM", model_name)
    print("model_path", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to('mps')
    model.eval()
    models[model_name] = model
    tokenizers[model_name] = tokenizer
    return model, tokenizer


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.95
    top_p: Optional[float] = 0.7
    max_length: Optional[int] = 2048
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get("/v1/models", response_model=ModelList)
async def list_models(token: str = Depends(verify_token)):
    global all_models
    data = []
    for model_name in all_models:
        model_card = ModelCard(id=model_name)
        data.append(model_card)
    return ModelList(data=data)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, token: str = Depends(verify_token)):
    global models, tokenizers
    model_name = request.model
    if model_name not in all_models:
        raise HTTPException(status_code=400, detail="Invalid model")
    model = models[model_name]
    tokenizer = tokenizers[model_name]
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i + 1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i + 1].content])

    if request.stream:
        generate = predict(request, model, tokenizer, query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history, max_length=request.max_length, top_p=request.top_p,
                             temperature=request.temperature)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(request: ChatCompletionRequest, model, tokenizer, query: str, history: List[List[str]],
                  model_id: str):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True, exclude_none=True))

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history, max_length=request.max_length,
                                             top_p=request.top_p, temperature=request.temperature):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True, exclude_none=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True, exclude_none=True))
    yield '[DONE]'


if __name__ == "__main__":
    for model_name in all_models:
        load_model(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model, trust_remote_code=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus(model, num_gpus=1)

    uvicorn.run(app, host='0.0.0.0', port=8282, workers=1)
