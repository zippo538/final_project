from src.services.mysql_store import MySQLChatMessageStore


def create_store(session_id: str = "default") -> MySQLChatMessageStore:
    return MySQLChatMessageStore(session_id=session_id)


# Default module-level store for backward compat
chat_store = MySQLChatMessageStore(session_id="default")
