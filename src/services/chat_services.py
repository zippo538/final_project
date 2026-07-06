from haystack.dataclasses import ChatMessage


class ChatLLMService:
    def __init__(self, predictor_pipeline, chat_history_pipeline, store, persist_category: bool = False):
        self.store = store
        self.predictor_pipeline = predictor_pipeline
        self.chat_history_pipeline = chat_history_pipeline
        self.persist_category = persist_category

    def _save_message(self, msg: ChatMessage, category: int = None):
        """Persist one message. When persist_category is True and category provided,
        use the category-tagged insert; otherwise fall back to default write."""
        if self.persist_category and category is not None:
            self.store.write_messages_with_category([msg], category=category)
        else:
            self.store.write_messages([msg])

    def process_chat(self, query: str, sentiment: str, category: int = None) -> str:
        # 1. Retrieve history from store
        history = self.chat_history_pipeline.run()

        # 2. Build messages
        user_msg = ChatMessage.from_user(query)

        # 3. Save user message (with category if configured)
        self._save_message(user_msg, category=category)

        # 4. Build context
        context = (history + "\nUser: " + query) if history else ("User: " + query)

        # 5. Run LLM
        response = self.predictor_pipeline.run(query, sentiment, context)
        response_text = response[0]._content[0].text

        # 6. Save assistant message (mirror category so the turn stays paired)
        assistant_msg = ChatMessage.from_assistant(response_text)
        self._save_message(assistant_msg, category=category)

        # 7. Return for endpoint
        return response_text, history
