from typing import Any, Dict, Iterable, List
import json
import pymysql
from pymysql.cursors import DictCursor

from haystack.dataclasses import ChatMessage
from haystack_experimental.chat_message_stores.types import ChatMessageStore

from src.utils.logger import default_logger as logger
from src.utils.config import config


class MySQLChatMessageStore(ChatMessageStore):
    def __init__(
        self,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        database: str = None,
        table: str = None,
        session_id: str = "default",
    ):
        self.host = host or config.get("mysql.host", "localhost")
        self.port = port or int(config.get("mysql.port", 3306))
        self.user = user or config.get("mysql.user", "root")
        self.password = password or config.get("mysql.password", "password")
        self.database = database or config.get("mysql.database", "chatbot_modi")
        self.table = table or config.get("mysql.table", "chat_messages")
        self.session_id = session_id
        self.connection = None
        self._connect()
        self._create_table()

    def _connect(self):
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                cursorclass=DictCursor,
                autocommit=True,
            )
            logger.info(f"MySQL connected: {self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"MySQL connection failed: {e}")
            raise

    def _ensure_connection(self):
        if self.connection is None or not self.connection.open:
            self._connect()

    def _create_table(self):
        self._ensure_connection()
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(128) NOT NULL,
            role VARCHAR(32) NOT NULL,
            content TEXT,
            meta JSON,
            category INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session (session_id),
            INDEX idx_created (created_at),
            INDEX idx_category (category)
        )
        """
        with self.connection.cursor() as cursor:
            cursor.execute(create_sql)
        logger.info(f"Table '{self.table}' ready")

    def _reconstruct_message(self, row: dict) -> ChatMessage:
        role = row["role"]
        content = row["content"] or ""
        meta = row.get("meta")
        category = row.get("category")
        if isinstance(meta, str):
            meta = json.loads(meta)
        meta = meta or {}

        # Preserve category in meta so callers can see sentiment it was tagged with
        if category is not None and "category" not in meta:
            meta["category"] = category

        kwargs = {"meta": meta} if meta else {}
        if role == "user":
            return ChatMessage.from_user(content, **kwargs)
        elif role == "assistant":
            return ChatMessage.from_assistant(content, **kwargs)
        elif role == "system":
            return ChatMessage.from_system(content, **kwargs)
        else:
            return ChatMessage.from_user(content, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "src.services.mysql_store.MySQLChatMessageStore",
            "init_parameters": {
                "host": self.host,
                "port": self.port,
                "user": self.user,
                "password": self.password,
                "database": self.database,
                "table": self.table,
                "session_id": self.session_id,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MySQLChatMessageStore":
        params = data.get("init_parameters", {})
        return cls(**params)

    def count_messages(self) -> int:
        self._ensure_connection()
        sql = f"SELECT COUNT(*) AS cnt FROM {self.table} WHERE session_id = %s"
        with self.connection.cursor() as cursor:
            cursor.execute(sql, (self.session_id,))
            row = cursor.fetchone()
            return row.get("cnt") or (row[0] if isinstance(row, tuple) else 0)

    def write_messages(self, messages: List[ChatMessage]) -> int:
        if not isinstance(messages, Iterable) or any(
            not isinstance(message, ChatMessage) for message in messages
        ):
            raise ValueError("Please provide a list of ChatMessages.")

        self._ensure_connection()
        inserted = 0
        with self.connection.cursor() as cursor:
            for msg in messages:
                role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                content = msg.text
                meta = json.dumps(msg.meta) if msg.meta else None
                sql = (
                    f"INSERT INTO {self.table} (session_id, role, content, meta) "
                    "VALUES (%s, %s, %s, %s)"
                )
                cursor.execute(sql, (self.session_id, role, content, meta))
                inserted += 1
        return inserted

    def write_messages_with_category(self, messages: List[ChatMessage], category: int) -> int:
        """Insert messages with a shared category (sentiment) value.

        Useful for recording a user question and the assistant's reply under
        the same sentiment score computed at request time.
        """
        if not isinstance(messages, Iterable) or any(
            not isinstance(message, ChatMessage) for message in messages
        ):
            raise ValueError("Please provide a list of ChatMessages.")

        self._ensure_connection()
        inserted = 0
        with self.connection.cursor() as cursor:
            for msg in messages:
                role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                content = msg.text
                meta = json.dumps(msg.meta) if msg.meta else None
                sql = (
                    f"INSERT INTO {self.table} (session_id, role, content, meta, category) "
                    "VALUES (%s, %s, %s, %s, %s)"
                )
                cursor.execute(sql, (self.session_id, role, content, meta, category))
                inserted += 1
        return inserted

    def delete_messages(self) -> None:
        self._ensure_connection()
        sql = f"DELETE FROM {self.table} WHERE session_id = %s"
        with self.connection.cursor() as cursor:
            cursor.execute(sql, (self.session_id,))
        logger.info(f"Deleted all messages for session '{self.session_id}'")

    def retrieve(self, limit: int = None) -> List[ChatMessage]:
        self._ensure_connection()
        if limit is None or limit <= 0:
            sql = (
                f"SELECT * FROM {self.table} "
                f"WHERE session_id = %s ORDER BY created_at ASC"
            )
            params = (self.session_id,)
        else:
            # Fetch latest N messages first (ORDER BY DESC + LIMIT), then flip to chronological order
            sql = (
                f"SELECT * FROM ("
                f"  SELECT * FROM {self.table} "
                f"  WHERE session_id = %s "
                f"  ORDER BY created_at DESC LIMIT %s"
                f") AS recent ORDER BY created_at ASC"
            )
            params = (self.session_id, limit)

        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [self._reconstruct_message(row) for row in rows]

    def close(self):
        if self.connection and self.connection.open:
            self.connection.close()
            logger.info("MySQL connection closed")
