import requests
import streamlit as st


BACKEND_BASE = "http://backend:8000"


st.set_page_config(page_title="Stock Sentiment Demo", page_icon="📈")

st.title("Russian Stock Sentiment")

mode = st.radio(
    "Режим работы сервиса",
    ["Одиночный текст", "Batch (forward_batch)", "Оценка датасета (evaluate)"],
)

if mode == "Одиночный текст":
    st.write("Введите текст новости/сообщения, нажмите **Predict**.")

    text = st.text_area("Текст для анализа", height=200)

    if st.button("Predict"):
        if not text.strip():
            st.warning("Введите текст.")
        else:
            with st.spinner("Модель думает..."):
                try:
                    resp = requests.post(f"{BACKEND_BASE}/forward", json={"text": text})
                    if resp.status_code == 200:
                        data = resp.json()
                        label = data.get("label")
                        probs = data.get("probs", [])
                        st.success(f"Предсказанный класс: {label}")
                        st.json({"label": label, "probs": probs})
                    elif resp.status_code == 400:
                        st.error("Bad request (400). Проверьте формат запроса.")
                    elif resp.status_code == 403:
                        st.error("Модель не смогла обработать данные (403).")
                    else:
                        st.error(f"Ошибка сервера: {resp.status_code}")
                except Exception as e:
                    st.error(f"Ошибка при обращении к backend: {e}")

elif mode == "Batch (forward_batch)":
    st.write("Загрузите `.csv` файл с колонкой `text`.")
    file = st.file_uploader("CSV файл для batch-инференса", type=["csv"])

    if st.button("Запустить forward_batch"):
        if file is None:
            st.warning("Сначала загрузите CSV файл.")
        else:
            with st.spinner("Модель обрабатывает batch..."):
                try:
                    files = {"file": (file.name, file.getvalue(), "text/csv")}
                    resp = requests.post(f"{BACKEND_BASE}/forward_batch", files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success("Batch-инференс выполнен успешно.")
                        st.json(data)
                    elif resp.status_code == 400:
                        st.error(f"Bad request (400): {resp.text}")
                    elif resp.status_code == 403:
                        st.error("Модель не смогла обработать данные (403).")
                    else:
                        st.error(f"Ошибка сервера: {resp.status_code}")
                except Exception as e:
                    st.error(f"Ошибка при обращении к backend: {e}")

elif mode == "Оценка датасета (evaluate)":
    st.write(
        "Загрузите `.csv` файл с колонками `text` и `target` (классы 0, 1, 2) "
        "для расчёта accuracy, precision, recall."
    )
    file = st.file_uploader("CSV файл для оценки модели", type=["csv"])

    if st.button("Запустить evaluate"):
        if file is None:
            st.warning("Сначала загрузите CSV файл.")
        else:
            with st.spinner("Модель оценивает датасет..."):
                try:
                    files = {"file": (file.name, file.getvalue(), "text/csv")}
                    resp = requests.post(f"{BACKEND_BASE}/evaluate", files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        metrics = data.get("metrics", {})
                        st.success("Оценка выполнена успешно.")
                        st.subheader("Метрики")
                        st.json(metrics)
                        st.subheader("Примеры предсказаний")
                        st.json(data.get("items", []))
                    elif resp.status_code == 400:
                        st.error(f"Bad request (400): {resp.text}")
                    elif resp.status_code == 403:
                        st.error("Модель не смогла обработать данные (403).")
                    else:
                        st.error(f"Ошибка сервера: {resp.status_code}")
                except Exception as e:
                    st.error(f"Ошибка при обращении к backend: {e}")

