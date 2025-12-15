import requests
import streamlit as st


BACKEND_URL = "http://backend:8000/forward"


st.set_page_config(page_title="Stock Sentiment Demo", page_icon="📈")

st.title("Russian Stock Sentiment")
st.write("Введите текст новости/сообщения, нажмите **Predict**.")

text = st.text_area("Текст для анализа", height=200)

if st.button("Predict"):
    if not text.strip():
        st.warning("Введите текст.")
    else:
        with st.spinner("Модель думает..."):
            try:
                resp = requests.post(BACKEND_URL, json={"text": text})
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


