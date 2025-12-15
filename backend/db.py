import os
import time
from datetime import datetime
from typing import List, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session


DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "features")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


class Base(DeclarativeBase):
    pass


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=True)
    label = Column(Integer, nullable=True)
    probs = Column(JSON, nullable=True)
    has_image = Column(Boolean, default=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,  # Проверка соединения перед использованием
    connect_args={"connect_timeout": 10},
)


def init_db() -> None:
    """Создание таблиц при старте приложения с retry логикой."""
    max_retries = 10
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Проверяем подключение к БД
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            # Если подключение успешно, создаем таблицы
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"БД недоступна, попытка {attempt + 1}/{max_retries}. Повтор через {retry_delay}с...")
                time.sleep(retry_delay)
            else:
                print(f"Не удалось подключиться к БД после {max_retries} попыток: {e}")
                raise


def log_inference(
    text: Optional[str],
    label: Optional[int],
    probs: Optional[List[float]],
    has_image: bool,
    status: str,
) -> None:
    """Лог инференса в таблицу inference_logs. Ошибки не пробрасываются наружу."""
    try:
        with Session(engine) as session:
            entry = InferenceLog(
                text=text,
                label=label,
                probs=probs,
                has_image=has_image,
                status=status,
            )
            session.add(entry)
            session.commit()
    except Exception:
        # Логирование БД не должно ломать основной сервис
        pass


