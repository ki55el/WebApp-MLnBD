import streamlit as st
import pandas as pd


df = pd.read_csv('data/neo_task.csv')
df_upd = pd.read_csv('data/neo_task_upd.csv')

st.set_page_config(page_title='Dataset', page_icon='💾')

st.markdown('''
# Информация о наборе данных

## Тематика
Этот датасет содержит различные признаки, на основе которых конкретный астероид, 
который уже классифицирован как ближайший к Земле объект, может быть или не быть опасным.

## Описание признаков
- `id` - уникальный идентификатор для каждого астероида
- `name` - имя, данное НАСА
- `est_diameter_min` - минимальный расчетный диаметр в километрах
- `est_diameter_max` - максимальный расчетный диаметр в километрах
- `relative_velocity` - скорость относительно Земли
- `miss_distance` - пропущенное расстояние в километрах
- `absolute_magnitude` - описывает собственную яркость
- `hazardous` - логический признак, показывающий, опасен астероид или нет

**Датасет до предобработки 👇**
''')
st.dataframe(df)

st.markdown('''
## Особенности предобработки данных
- Удалены столбцы `id` и `name`
```
df.drop(['id','name'], axis=1, inplace=True)
```
- Изменён тип данных столбца `hazardous` с bool на int
```
df['hazardous'] = df['hazardous'].astype(int)
```
- Удалены строки, в которых отсутствует хотя бы один элемент
```
df.dropna(inplace=True)
```
- Удалены выбросы путем определения нижнего/верхнего предела нормального диапазона значений
```
iX = iX = df.drop(['hazardous'], axis=1).columns

Q1 = df[iX].quantile(0.25)
Q3 = df[iX].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df[iX] = np.clip(df[iX], lower, upper, axis=1)
```

**Датасет после предобработки 👇**
''')
st.dataframe(df_upd)
