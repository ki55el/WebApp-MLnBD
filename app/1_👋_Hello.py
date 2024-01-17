import streamlit as st


st.set_page_config(page_title='Hello', page_icon='👋')

st.markdown('''
# Web-приложение для вывода моделей ML и анализа данных\n
# Информация о разработчике
''')

col1, col2 = st.columns(2)

with col1:
    st.markdown('''
## ФИО: Сыздыков Дамир Алматович\n
## Номер учебной группы: МО-221
''')

with col2:
    st.image("static/329.jpg", width=256)
