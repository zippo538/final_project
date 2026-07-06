from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import ChatMessage
from typing import List
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from src.utils.logger import default_logger as logger
from src.utils.config import config
import os
import json
import mlflow
import requests
load_dotenv()




@component
class GroqLLM:
    def __init__(self, model_name : str , api_key=None):
        self.api_key = api_key or os.getenv("OPEN_ROUTER_API_KEY")
        self.model_name = model_name or os.getenv("MODEL_AI")

    @component.output_types(output=List[ChatMessage])
    def run(self, prompt: List[ChatMessage]):
        # Extract user messages properly
        user_prompt = "".join([msg.text for msg in prompt])

        url = os.getenv("URL_API")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_prompt.strip()}],
            "temperature": 1,
            "max_tokens": 300
        }

        response = requests.post(url, headers=headers, json=payload)
        # Check response status
        if response.status_code != 200:
            raise ValueError(
                f"OpenRouter API returned status {response.status_code}.\n"
                f"Response: {response.text}"
            )
        try:
            data = response.json()
        except Exception:
            raise ValueError("Gagal parse JSON dari Groq API: ", response.text)
        
        # Debug: print response structure
        logger.debug(f"OpenRouter API Response: {json.dumps(data, indent=2)}")

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
        self.pipeline.add_component('prompt_builder',PromptBuilder(
            template=config.get("prompt.predictor_prompt"),
            variables={"sentiment", "input", "context"},
            required_variables={"sentiment", "input"},
        ))
        self.pipeline.add_component('prompt_to_msg',PromptToMessages())
        self.pipeline.add_component('groq_llm',GroqLLM())
        
        self.pipeline.connect("prompt_builder.prompt","prompt_to_msg.prompt")
        self.pipeline.connect("prompt_to_msg.messages","groq_llm.prompt")
    
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
    def __init__(self, chat_message_store, limit: int = 20):
        self.chat_message_store = chat_message_store
        self.limit = limit

    def run(self):
        # Fetch recent messages directly. MySQLChatMessageStore honors `limit`;
        # Haystack's InMemoryChatMessageStore does not, so fallback + slice.
        try:
            messages = self.chat_message_store.retrieve(limit=self.limit)
        except (TypeError, AttributeError):
            messages = self.chat_message_store.retrieve()
            if self.limit and self.limit > 0 and len(messages) > self.limit:
                messages = messages[-self.limit:]
        if not messages:
            return ""

        lines = []
        role_labels = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        }
        for msg in messages:
            role_value = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            label = role_labels.get(role_value, "User")
            text = (msg.text or "").strip()
            if text:
                lines.append(f"{label}: {text}")
        return "\n".join(lines)

        

    
    
        