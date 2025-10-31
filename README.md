# 🧠 Inventory Demand Forecasting

This is a simple machine learning project where I built models to predict the future demand for retail products based on past inventory and sales data.  
The main idea is to help stores maintain the right stock levels — avoiding both overstocking and stockouts.

---

## 📊 Project Summary
Retail management often struggles to decide **how much stock to order**.  
Using historical sales data, product categories, pricing, and other factors like promotions or weather, this project predicts **future product demand** with the help of regression models.

---

## ⚙️ Tools and Technologies
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Models:** Ridge, Lasso, Decision Tree, Random Forest, AdaBoost, Gradient Boosting  
- **Environment:** Jupyter Notebook (Anaconda)  
- **Version Control:** Git and GitHub  

---

## 🧩 Dataset
The dataset contains details such as:
- Store ID, Product ID, Category, and Region  
- Inventory level, Units sold, and Units ordered  
- Demand forecast, Price, and Discount  
- Weather conditions, Promotions, Competitor pricing, and Seasonality  

**Target column:** `demand forecast`  
The goal is to train models that can accurately predict this target value.

---

## 🔍 Workflow
1. **Data Cleaning & Preprocessing**
   - Converted date fields into year, month, and day  
   - Encoded categorical data using LabelEncoder and OneHotEncoder  
   - Scaled numeric columns with StandardScaler  

2. **Model Training**
   - Tried out different regression algorithms (Ridge, Lasso, Decision Tree, Random Forest, AdaBoost, Gradient Boosting)  
   - Evaluated each model using R² Score and RMSE  

3. **Hyperparameter Tuning**
   - Used `RandomizedSearchCV` to find the best parameters for Ridge, Lasso, AdaBoost, and Gradient Boosting  

4. **Model Comparison**
   - Compared results to see which model performed best on unseen data  

---

## 🧠 Results

| Model | R² Score | RMSE |
|--------|-----------|-------|
| Ridge Regression | 0.957 | 514 |
| Lasso Regression | 0.948 | 580 |
| Decision Tree | 0.912 | 720 |
| Random Forest | **0.961** | **470** |
| AdaBoost | 0.943 | 600 |
| Gradient Boosting | 0.956 | 520 |

🟢 **Best Model:** Random Forest Regressor  
It gave the highest R² score and the lowest RMSE among all models.

---

## 💻 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<yourusername>/inventory_demand_forecasting.git
