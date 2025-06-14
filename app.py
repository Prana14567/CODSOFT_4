import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("📈 Sales Prediction App")
st.write("Predict product sales based on advertising budgets using Random Forest Regression.")



X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.sidebar.header("Input Advertising Budgets")
tv_input = st.sidebar.slider("TV Budget (₹)", 0, 300, 100)
radio_input = st.sidebar.slider("Radio Budget (₹)", 0, 60, 20)
newspaper_input = st.sidebar.slider("Newspaper Budget (₹)", 0, 100, 30)


user_input = np.array([[tv_input, radio_input, newspaper_input]])


predicted_sales = model.predict(user_input)[0]

st.subheader("💰 Predicted Sales:")
st.success(f"{predicted_sales:.2f} units")


st.subheader("🔍 Feature Importances:")
importance_df = pd.DataFrame({
    'Feature': ['TV', 'Radio', 'Newspaper'],
    'Importance': model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index('Feature'))
