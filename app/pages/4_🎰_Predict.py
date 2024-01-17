import pickle

from imblearn.over_sampling import SMOTE
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import streamlit as st


def get_metrics(actual, pred):
  return {
    'Accuracy': accuracy_score(actual, pred),
    'Precision': precision_score(actual, pred),
    'Recall': recall_score(actual, pred),
    'F1-score': f1_score(actual, pred),
    'ROC_AUC': roc_auc_score(actual, pred)
  }

df = pd.read_csv('data/neo_task_upd.csv')
y = df['hazardous']
X = df.drop(['hazardous'], axis=1)

with open("models/KNeighborsClassifier.pkl", "rb") as f:
    model1 = pickle.load(f)
with open("models/KMeans.pkl", "rb") as f:
    model2 = pickle.load(f)
with open("models/GradientBoostingClassifier.pkl", "rb") as f:
    model3 = pickle.load(f)
with open("models/BaggingClassifier.pkl", "rb") as f:
    model4 = pickle.load(f)
model5 = load_model('models/NN.h5')

models = {
    'KNeighborsClassifier': model1,
    'KMeans': model2,
    'GradientBoostingClassifier': model3,
    'BaggingClassifier': model4,
    'Neural Network': model5 
}

st.set_page_config(page_title='Predict', page_icon='🎰')

st.write('# Предсказание моделей машинного обучения')

uploaded_file = st.file_uploader('__Загрузите файл в формате *.csv__', type='csv')

if uploaded_file:
    df_inp = pd.read_csv(uploaded_file)
    st.write('## Для следующего датасета:', df_inp)
    y_inp = df_inp['hazardous']
    X_inp = df_inp.drop(['hazardous'], axis=1)
    
    smote = SMOTE()
    X_inp, y_inp = smote.fit_resample(X_inp, y_inp)

    scaler = MinMaxScaler()
    X_inp = scaler.fit_transform(X_inp)

    predicts = {}
    for name, model in models.items():
        predicts[name] = model.predict(X_inp)
    predicts['Neural Network'] = np.ravel(np.around(predicts['Neural Network']).astype(np.int64))

    st.write(
        '## Получаем следующие предсказания:', 
        pd.DataFrame(data=predicts),
        '## И следующие метрики:'
        )
    for name, pred in predicts.items():
        st.write(
            f'- ### `{name}`:',
            get_metrics(y_inp, pred)
            )

else:
    st.write('__Или введите данные для предсказания 👇__')

    inp = {}
    for ix in X:
        inp[ix] = st.slider(f'**{ix}=**', min(X[ix]), max(X[ix]))
    
    X_inp = pd.DataFrame([inp])
    st.write(
        '## Для следующего набора данных:', 
        X_inp, 
        '## Получаем следующие предсказания:'
        )

    for name, model in models.items():
        st.write(f'### `{name}`: ', 'астероид опасен ☠️' if model.predict(X_inp) else 'астрероид безопасен ☮️')
