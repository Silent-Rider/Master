import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import service

notification_shown = False

st.set_page_config(page_title="Прогнозирование цены бриллианта", layout="wide")
st.title("💎 Прогнозирование цены бриллианта")

service.init_models_and_scalers()

linear_model = st.session_state.linear_model
keras_model = st.session_state.keras_model
scaler_x:MinMaxScaler = st.session_state.scaler_x
scaler_y:MinMaxScaler = st.session_state.scaler_y

input_column, margin, output_column = st.columns([2, 0.25, 3])

with input_column:
    ### Входные данные
    st.header("Характеристика алмаза")

    carat = st.slider("Караты", min_value=0.2, max_value=2.0, value=1.1, step=0.01)
    cut = st.selectbox("Качество огранки", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=2)
    color = st.radio("Цвет", ['J', 'I', 'H', 'G', 'F', 'E', 'D'], index=3, horizontal=True)
    clarity = st.selectbox("Чистота", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], index=4)
    depth = st.slider("Глубина", min_value=58.8, max_value=64.7, value=61.8, step=0.1)
    table = st.slider("Плоская грань", min_value=52.0, max_value=63.5, value=57.8, step=0.1)
    x = st.slider("Длина", min_value=3.73, max_value=8.28, value=6.0, step=0.01)
    y = st.slider("Высота", min_value=3.68, max_value=8.27, value=5.98, step=0.01)
    z = st.slider("Ширина", min_value=1.41, max_value=5.3, value=3.36, step=0.01)

    ### Кнопка предсказания
    st.markdown("""
    <style>
        button[kind="primary"] {
            background: linear-gradient(135deg, #8b5cf6, #7c3aed);
            color: white;
            border-radius: 12px;
            padding: 16px 28px;
            font-size: 20px;
            font-weight: 600;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.3);
        }
        button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

    clicked = st.button("Предсказать", type="primary", use_container_width=True)

with output_column:
    input_data = {
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    }

    ### Таблица исходных X
    st.markdown(
        "<h2 style='text-align: center; color: #2d3748; font-size: 30px';>исходные X</h2>",
        unsafe_allow_html=True
    )
    df_raw_x = pd.DataFrame(input_data)
    st.dataframe(
        df_raw_x,
        hide_index=True,
        use_container_width=True,
        column_config={
            "carat": st.column_config.NumberColumn("carat", format="%.2f"),
            "depth": st.column_config.NumberColumn("depth", format="%.1f"),
            "table": st.column_config.NumberColumn("table", format="%.1f"),
            "x": st.column_config.NumberColumn("x", format="%.2f"),
            "y": st.column_config.NumberColumn("y", format="%.2f"),
            "z": st.column_config.NumberColumn("z", format="%.2f"),
        }
    )

    ### Таблица нормализованных X
    df_x = service.get_prepared_df_x(df_raw_x)
    st.markdown(
        "<h2 style='text-align: center; color: #2d3748; font-size: 30px';>нормализованные X</h2>",
        unsafe_allow_html=True
    )
    columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    df_norm_x = pd.DataFrame (data = scaler_x.transform(df_x), columns = df_x.columns, index = df_x.index)
    st.dataframe(df_norm_x, hide_index=True, use_container_width=True)

    col_linear, col_neural = st.columns(2)

    if clicked:
        with col_linear:
            st.header("Линейная регрессия")
            st.write("**R²=0.9163**")
            st.write("**RMSE=797.49**")

            st.subheader("предсказанный Y")
            pred_price_linear = max(0, linear_model.predict(df_x).item())

            st.metric(label="price", value=f"${pred_price_linear:,.2f}")

        with col_neural:
            st.header("Нейронная сеть")
            st.write("**R²=0.974**")
            st.write("**RMSE=0.0353**")

            st.subheader("нормализованный Y")
            norm_y_neural = max(0, keras_model.predict(df_norm_x).item())
            st.metric(label="price", value=f"{norm_y_neural:.4f}")

            st.subheader("предсказанный Y")
            pred_price_neural = scaler_y.inverse_transform([[norm_y_neural]]).item()
            st.metric(label="price", value=f"${pred_price_neural:,.2f}")

st.markdown("---")