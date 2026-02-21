import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
from tensorflow.keras.models import load_model

FOLDER = 'models/'

def init_models_and_scalers():
    if 'models_loaded' not in st.session_state:
        try:
            linear_model, keras_model, scaler_x, scaler_y = load_models_and_scalers()
            st.session_state.update({
                'linear_model': linear_model,
                'keras_model': keras_model,
                'scaler_x': scaler_x,
                'scaler_y': scaler_y,
                'models_loaded': True
            })
            st.toast("✅ Модели и шкалёры успешно загружены.")
        except Exception as e:
            st.toast(f"❌ Ошибка загрузки: {e}")
            st.stop()


@st.cache_resource
def load_models_and_scalers():
    with open(FOLDER + 'linear_model.pkl', 'rb') as f:
        lin_model = pickle.load(f)

    ker_model = load_model(FOLDER + 'keras_model.h5', compile=False)

    with open(FOLDER + 'scaler_x.pkl', 'rb') as f:
        scal_x = pickle.load(f)

    with open(FOLDER + 'scaler_y.pkl', 'rb') as f:
        scal_y = pickle.load(f)

    return lin_model, ker_model, scal_x, scal_y


def get_prepared_df_x(df) -> DataFrame:
    df = df.copy()
    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    df['color'] = df['color'].map({v: i + 1 for i, v in enumerate(color_order)})

    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    df['cut'] = df['cut'].map({v: i + 1 for i, v in enumerate(cut_order)})

    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    df['clarity'] = df['clarity'].map({v: i + 1 for i, v in enumerate(clarity_order)})
    return df
