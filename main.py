import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

import matplotlib.pyplot as plt
import plotly.express as px

movies = pd.read_csv('tmdb_5000_movies.csv')

st.write("## Exploratory Data Analysis (EDA)")

df = movies[['budget', 'runtime', 'genres', 'revenue']]
df.dropna(subset=['budget', 'runtime', 'revenue'], inplace=True)
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

df['genres'] = df['genres'].apply(lambda x: [genre['name'] for genre in eval(x)])
unique_genres = set([genre for sublist in df['genres'] for genre in sublist])

for genre in unique_genres:
    df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)

df.drop('genres', axis=1, inplace=True)

st.write("### Distribution of Movie Revenues")
fig_revenue = px.histogram(df, x='revenue', nbins=50, title="Distribution of Revenues")
st.plotly_chart(fig_revenue)

st.write("### Budget vs Revenue")
fig_budget_revenue = px.scatter(df, x='budget', y='revenue', title="Budget vs Revenue",
                                labels={'budget': 'Budget', 'revenue': 'Revenue'},
                                trendline="ols")
st.plotly_chart(fig_budget_revenue)

st.write("### Movie Runtime Distribution")
fig_runtime = px.histogram(df, x='runtime', nbins=30, title="Distribution of Runtime (Minutes)")
st.plotly_chart(fig_runtime)

st.write("### Top Genres by Average Revenue")
genre_revenue = pd.DataFrame({genre: df[df[genre] == 1]['revenue'].mean() for genre in unique_genres}, index=['avg_revenue']).T
genre_revenue = genre_revenue.sort_values(by='avg_revenue', ascending=False)
fig_genre_revenue = px.bar(genre_revenue, x=genre_revenue.index, y='avg_revenue', 
                           title="Top Genres by Average Revenue", labels={'index': 'Genre', 'avg_revenue': 'Average Revenue'})
st.plotly_chart(fig_genre_revenue)

st.write("### Correlation Heatmap of Features")
corr = df.corr()
fig_corr = plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(fig_corr)

X = df.drop('revenue', axis=1)
y = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("## Model Evaluation")

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f'**R² score**: {r2}')
st.write(f'**Root Mean Squared Error (RMSE)**: {rmse}')

st.write("### Actual vs Predicted Revenue")
fig_actual_vs_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                 title="Actual vs Predicted Revenue")
fig_actual_vs_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
              line=dict(color='Red',))
st.plotly_chart(fig_actual_vs_pred)

st.write("## Feature Importance")

coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_
})
coefficients['abs_coeff'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='abs_coeff', ascending=False)

st.bar_chart(coefficients.set_index('Feature')['Coefficient'])

st.write("## Predict Movie Box Office Revenue")

budget_input = st.number_input('Enter Movie Budget', min_value=1, value=1000000)
runtime_input = st.number_input('Enter Movie Runtime (minutes)', min_value=1, value=120)

genre_columns = X.columns[2:]
selected_genres = st.multiselect('Select Movie Genres', genre_columns)

user_input = np.zeros(len(X_train.columns))
user_input[0] = budget_input
user_input[1] = runtime_input
for genre in selected_genres:
    idx = X.columns.get_loc(genre)
    user_input[idx] = 1

if st.button('Predict Box Office Revenue'):
    user_input = user_input.reshape(1, -1)
    prediction = model.predict(user_input)
    st.write(f"Predicted Box Office Revenue: ${prediction[0]:,.2f}")
