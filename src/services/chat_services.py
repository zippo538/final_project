from haystack.dataclasses import ChatMessage
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.writers import ChatMessageWriter

class ChatLLMService:
    def __init__(self, predictor_pipeline, chat_history_pipeline):
        # singleton store
        self.store = InMemoryChatMessageStore()
        self.writer = ChatMessageWriter(self.store)
        self.predictor_pipeline = predictor_pipeline
        self.chat_history_pipeline = chat_history_pipeline

    def process_chat(self, query: str, sentiment:str, context) -> str:
        # 1. Ambil history
        history = self.chat_history_pipeline.run()
        history_text = history["messages"] if "messages" in history else ""

        # 2. Buat ChatMessage
        user_msg = ChatMessage.from_user(query)
        history_msg = ChatMessage.from_system(history_text)

        # 3. Simpan user ke store
        self.writer.run([user_msg])

        # 4. Jalankan LLM
        response = self.predictor_pipeline.run(query,sentiment, context)
        response_text = response[0]._content[0].text

        # 5. Simpan assistant ke store
        assistant_msg = ChatMessage.from_assistant(response_text)
        self.writer.run([assistant_msg])

        # 6. Return untuk endpoint
        return response_text
