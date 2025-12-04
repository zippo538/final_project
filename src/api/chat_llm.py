from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack.dataclasses import ChatMessage
from typing import List
from haystack.components.builders import ChatPromptBuilder
from haystack.components.joiners import ListJoiner
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from src.utils.logger import default_logger as logger
from src.utils.config import config
import os
import json
import mlflow
import requests
load_dotenv()



chat_message_store = InMemoryChatMessageStore()

@component
class GroqLLM:
    def __init__(self, model_name="meta-llama/llama-4-maverick-17b-128e-instruct", api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name

    @component.output_types(output=List[ChatMessage])
    def run(self, prompt: List[ChatMessage]):
        user_prompt = "".join([msg.text for msg in prompt])
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.7,
            "max_tokens": 300
        }

        response = requests.post(url, headers=headers, json=payload)
        try:
            data = response.json()
        except Exception:
            raise ValueError("Gagal parse JSON dari Groq API: ", response.text)

        # Debug untuk melihat isi JSON asli
        if "choices" not in data:
            raise ValueError(
                "Groq API tidak mengembalikan 'choices'.\n"
                f"Status Code: {response.status_code}\n"
                f"Response JSON:\n{json.dumps(data, indent=2)}"
            )

        # Jika OK, ambil isi respon
        result = data["choices"][0]["message"]["content"]
        return {"output": [ChatMessage.from_assistant(result)]}
    
@component
class PredictorCategory:
    def __init__(self, model_name , model_tfidf):
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        self.client = MlflowClient()
        self.model = self._load_model(model_name)
        self.tfidf = self._load_model(model_tfidf)

    @component.output_types(category=str)
    def run(self, input_data: str):
        transform = self.tfidf.transform([input_data])
        category = self.model.predict(transform)
        return {"category": category[0]}
    
    def _load_model(self,model_name:str) :
        version = self.client.search_model_versions(f"name='{model_name}'")
        latest_version = max(version, key=lambda x:int(x.version))
        last_version_number = latest_version.version
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{last_version_number}")
        return model
    
@component
class PromptToMessages:
    @component.output_types(messages=list[ChatMessage])
    def run(self, prompt: str):
        # Convert string → List[ChatMessage]
        messages = [
            ChatMessage.from_user(prompt)
        ]
        return {"messages": messages}

class PipelineCategory:
    def __init__(self):
        
        self.pipeline = Pipeline()
        self.pipeline.add_component('prompt_builder',PromptBuilder(template=config.get("prompt.predictor_prompt"),required_variables=["sentiment","input","context"]))
        self.pipeline.add_component('prompt_to_msg',PromptToMessages())
        self.pipeline.add_component('groq_llm',GroqLLM())
        
        self.pipeline.connect("prompt_builder.prompt","prompt_to_msg.prompt")
        self.pipeline.connect("prompt_to_msg.messages","groq_llm")
    
    def run(self,input_text : str , sentiment : str, context) :
        res = self.pipeline.run(
            data={
                "prompt_builder" : {
                    "input" : input_text,
                    "sentiment" : sentiment,
                    "context" : context
                }
            })
        return res['groq_llm']['output']

class ChatHistoryPipeline:
    def __init__(self, chat_message_store):
        self.chat_message_store = chat_message_store
        self.pipeline = Pipeline()
        self.pipeline.add_component("memory_retriever", ChatMessageRetriever(chat_message_store))
        self.pipeline.add_component("prompt_builder", PromptBuilder(variables=["memories"], required_variables=["memories"], template="""
        Previous Conversations history:
        {% for memory in memories %}
            {{memory.text}}
        {% endfor %}
        """)
        )
        self.pipeline.connect("memory_retriever", "prompt_builder.memories")

    def run(self):
        res = self.pipeline.run(
            data = {},
            include_outputs_from=["prompt_builder"]
        )

        # print("Pipeline Input", res["prompt_builder"]["prompt"])
        return res["prompt_builder"]["prompt"]

        
chat_history_pipeline = ChatHistoryPipeline(chat_message_store=chat_message_store)
            
    
    
        