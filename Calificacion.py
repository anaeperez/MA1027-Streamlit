import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de la calificación  ''')
st.image("calif.jfif", caption="Predicción de la calificación de un examen.")

st.header('Datos del alumno')

def user_input_features():
  # Entrada
  Estudio = st.number_input('Horas de estudio:', min_value=1, max_value=12, value = 0, step = 1)
  Sueno = st.number_input('Horas de sueño:',  min_value=4, max_value=9, value = 0, step = 1)
  Asistencia = st.number_input('Porcentaje de asistencia:', min_value=50, max_value=100, value = 0, step = 1)
  Previo = st.number_input('Calificación previa:', min_value=40, max_value=95, value = 0, step = 1)

  user_input_data = {'hours_studied': Estudio,
                     'sleep_hours': Sueno,
                     'attendance_percent': Asistencia,
                     'previous_scores': Previo
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
prediccion=0
datos =  pd.read_csv('Calificación.csv', encoding='latin-1')
X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores']

st.subheader('Calificación del alumno')
st.write('La calificación que se predice es ', prediccion)
