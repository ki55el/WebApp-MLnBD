import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/neo_task_upd.csv')


def get_boxplot():
    st.write('## Ящик с усами')
    x = st.selectbox('**Выберите x=**', df.columns)
    plt.figure()    
    plt.boxplot(x=df[x])
    st.pyplot(plt)

def get_heatmap():
    st.write('## Тепловая карта')
    plt.figure()
    sns.heatmap(data=df.corr(numeric_only=True), annot=True)
    st.pyplot(plt) 

def get_hist():
    st.write('## Гистограмма')
    x = st.selectbox('**Выберите x=**', df.columns)
    plt.figure()    
    plt.hist(x=df[x])
    st.pyplot(plt)

def get_scatterplot():
    st.write('## Диаграмма рассеяния')
    x = st.selectbox('**Выберите x=**', df.columns, key='x')
    y = st.selectbox('**Выберите y=**', df.columns, key='y')
    plt.figure()    
    sns.scatterplot(x=df[x], y=df[y])
    st.pyplot(plt)


st.set_page_config(page_title='Visualization', page_icon='🎨')

st.markdown('''
# Визуализация данных
**Датасет для визуализации 👇**
''')
st.dataframe(df)

visualize = {
    'Ящик с усами': get_boxplot,
    'Тепловая карта': get_heatmap,
    'Гистограмма': get_hist,
    'Диаграмма рассеяния': get_scatterplot
}

plot = st.selectbox('**Выберите вид визуализации**', visualize.keys())
visualize[plot]()
