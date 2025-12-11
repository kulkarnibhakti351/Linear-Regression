📦 Food Delivery Time Prediction — Linear Regression Project
📊 Predicting Delivery Time Using Machine Learning (Python)

This project builds a predictive model to estimate food delivery time based on multiple factors such as distance, order preparation time, delivery partner speed, and time of day. It also includes detailed statistical analysis, model evaluation, and business recommendations.

🧠 Project Objectives

Explore and clean the dataset

Perform feature engineering & encoding

Check multicollinearity using VIF

Train a Linear Regression model

Predict delivery time for all inputs

Compute MAE, MSE, R²

Identify top features affecting delivery time

Provide actionable business recommendations

📁 Project Structure
📦 food-delivery-time-prediction
│
├── data/
│   └── delivery_data.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate.py
│
└── README.md

🔍 Steps Performed
1️⃣ Data Cleaning & Preprocessing

✔ Removed unnecessary columns
✔ Handled missing values
✔ Converted categorical data using One-Hot Encoding
✔ Scaled numerical features using MinMaxScaler
✔ Stored feature names before scaling (feature_names = x_train.columns)

2️⃣ Checking Multicollinearity (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(vif_data)

📌 Why VIF?

VIF shows how severe multicollinearity is.

VIF > 5 or 10 → Serious multicollinearity.

3️⃣ Model Training
Simple Linear Regression
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train, y_train)


✔ Coefficients extracted for feature importance
✔ Predictions stored for entire dataset

4️⃣ Predictions & New Columns
Q13 → Predict Delivery Time
predictions = regression.predict(x)

Q14 → Add Predicted Column
df['Predicted_Delivery_Time'] = predictions

Q15 → Calculate Residuals
df['Residual'] = df['Delivery_Time_min'] - df['Predicted_Delivery_Time']

5️⃣ Error Metrics
Q16 → Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions_test)

Q17 → Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions_test)

6️⃣ R² Score
Q18 → Coefficient of Determination
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions_test)


📌 Interpretation:
R² measures how much of the variation in delivery time is explained by the model.
Example:

R² = 0.82 → Model explains 82% of delivery time variation.

7️⃣ Feature Importance (Top 3)
coeff = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': regression.coef_
})

coeff['Absolute_Impact'] = coeff['Coefficient'].abs()
top3_features = coeff.sort_values(by='Absolute_Impact', ascending=False).head(3)


🟦 Features with strongest impact on delivery time
🟩 Based on absolute coefficient magnitude

🧩 Top 3 Features (Example)
Rank	Feature	Impact
1	Distance_km	⭐⭐⭐⭐⭐
2	Order_Preparation_Time	⭐⭐⭐⭐
3	Delivery_Partner_Speed	⭐⭐⭐
🏢 Business Recommendations (Q20)
1️⃣ Reduce Restaurant Preparation Time

Invest in kitchen process automation.

Prioritize items that take longer to prepare.

2️⃣ Optimize Delivery Partner Assignment

Assign closest delivery partner automatically.

Use speed history to match faster partners.

3️⃣ Reduce Travel Distance Impact

Use dynamic routing algorithms

Encourage restaurants to set optimal delivery radius

🛠 Technologies Used

Python

Pandas, NumPy

Scikit-Learn

Matplotlib / Seaborn

Jupyter Notebook

▶️ How to Run
pip install -r requirements.txt
python src/train_model.py
python src/evaluate.py

🤝 Contributing

Contributions are welcome!
You may fork the repository and create a pull request.

📄 License

MIT License
