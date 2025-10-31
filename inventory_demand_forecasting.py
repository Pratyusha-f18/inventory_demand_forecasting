#!/usr/bin/env python
# coding: utf-8


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean,median,mode


# In[6]:


df=pd.read_csv("C://Users//HP//Downloads//archive (7)//retail_store_inventory.csv")
df.head()


# In[8]:


df.tail()


# In[10]:


df.shape


# In[12]:


df.isnull().sum()


# In[14]:


df.dtypes


# In[16]:


df.info()


# In[18]:


df.describe()


# In[20]:


#finding outliers
df['Inventory Level'].plot(kind='box')


# In[22]:


df['Units Sold'].plot(kind='box')


# In[26]:


df['Units Ordered'].plot(kind='box')


# In[28]:


df['Competitor Pricing'].plot(kind='box')


# In[30]:


#handeling outliers
upper_limit_inventoryl=df['Inventory Level'].quantile(0.96)
print(f"\nUpper limit for IL: {upper_limit_inventoryl:.2f}")
df['Inventory Level']=np.where(df['Inventory Level']>upper_limit_inventoryl,upper_limit_inventoryl,df['Inventory Level'])
print('Outliers removed')
sns.boxplot(y=df['Inventory Level'])
plt.show()


# In[32]:


upper_limit_us=df['Units Sold'].quantile(0.90)
print(f"\nUpper limit for us: {upper_limit_us:.2f}")
df['Units Sold']=np.where(df['Units Sold']>upper_limit_us,upper_limit_us,df['Units Sold'])
print('Outliers removed')
sns.boxplot(y=df['Units Sold'])
plt.show()


# In[34]:


upper_limit_uo=df['Units Ordered'].quantile(0.90)
print(f"\nUpper limit for uo: {upper_limit_uo:.2f}")
df['Units Ordered']=np.where(df['Units Ordered']>upper_limit_uo,upper_limit_uo,df['Units Ordered'])
print('Outliers removed')
sns.boxplot(df['Units Ordered'])
plt.show()


# In[36]:


upper_limit_cp=df['Competitor Pricing'].quantile(0.99)
print(f"\nUpper limit for cp: {upper_limit_cp:.2f}")
df['Competitor Pricing']=np.where(df['Competitor Pricing']>upper_limit_cp,upper_limit_cp,df['Competitor Pricing'])
print('Outliers removed')
sns.boxplot(df['Competitor Pricing'])
plt.show()


# In[38]:


df['Price'].plot(kind='box')


# In[40]:


upper_limit_price=df['Price'].quantile(0.90)
print(f"\nUpper limit for price: {upper_limit_price:.2f}")
df['Price']=np.where(df['Price']>upper_limit_price,upper_limit_price,df['Price'])
print('Outliers removed')
sns.boxplot(df['Price'])
plt.show()


# In[42]:


df['Discount'].plot(kind='box')


# In[44]:


upper_limit_d=df['Discount'].quantile(0.83)
print(f"\nUpper limit for dis: {upper_limit_uo:.2f}")
df['Discount']=np.where(df['Discount']>upper_limit_d,upper_limit_d,df['Discount'])
print('Outliers removed')
sns.boxplot(df['Discount'])
plt.show()


# In[48]:


df.columns = df.columns.str.strip()         # remove spaces
df.columns = df.columns.str.lower()         # make lowercase


# In[52]:


df.columns = df.columns.str.strip()  # remove all leading/trailing spaces
print(df.columns.tolist())           # check again



# In[54]:


sns.histplot(df['demand forecast'], bins=30, kde=True)
plt.title("Distribution of Demand Forecast")
plt.xlabel("Demand Forecast")
plt.ylabel("Count")
plt.show()


# In[58]:


df['price'].unique()


# In[64]:


print(df.columns.tolist())



# In[66]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 1: Define which columns to encode
label_cols = ['store id', 'product id', 'seasonality']          # Label Encode
onehot_cols = ['category', 'region', 'weather condition']       # One-Hot Encode

# Step 2: Apply Label Encoding (for ordinal/high-cardinality columns)
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Step 3: Apply One-Hot Encoding (for nominal columns)
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False), onehot_cols)
    ],
    remainder='passthrough'  # Keeps all other columns
)

# Step 4: Apply transformation
encoded_array = ct.fit_transform(df)
df_encoded = pd.DataFrame(encoded_array)

# Step 5: Restore proper column names
onehot_feature_names = ct.named_transformers_['onehot'].get_feature_names_out(onehot_cols)
other_cols = [c for c in df.columns if c not in onehot_cols]
df_encoded.columns = list(onehot_feature_names) + other_cols

# Step 6: Reorder columns — move Year, Month, Day to the front
cols = df_encoded.columns.tolist()
date_cols = ['year', 'month', 'day']
new_order = date_cols + [c for c in cols if c not in date_cols]
df_encoded = df_encoded[new_order]

print("✅ Encoding complete & columns reordered!")
print("Final columns:", df_encoded.columns.tolist())


# In[68]:


df_encoded.head()


# In[70]:


plt.figure(figsize=(25,25))
sns.heatmap(df_encoded.corr(),annot=True,cmap='Blues')


# In[72]:


from sklearn.model_selection import train_test_split

# Separate features and target
X = df_encoded.drop('demand forecast', axis=1)
y = df_encoded['demand forecast']

# Split data into training and test sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Check shapes
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[76]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[78]:


X_train


# In[80]:


X_test


# In[82]:


#model training
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
reg.coef_


# In[84]:


reg.intercept_


# In[86]:


#prediction
y_pred=reg.predict(X_test)


# In[88]:


#metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print(mse)
print(mae)
print(score)


# ### HYPERPARAMETER TUNING
# 

# In[97]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
# Create the Ridge model
ridge = Ridge(random_state=42)
# Define a small parameter grid for fast execution
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}
# Initialize GridSearchCV
grid_ridge = GridSearchCV(
    estimator=ridge,
    param_grid=param_grid,
    cv=5,          # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1
)
# Fit on training data
grid_ridge.fit(X_train, y_train)
# Get the best model and alpha
best_ridge1 = grid_ridge.best_estimator_
best_alpha = grid_ridge.best_params_['alpha']
# Predict on test data
y_pred = best_ridge1.predict(X_test)
# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Display results
print("Best Alpha Value:", best_alpha)
print("Best Cross-Validation R2 Score:", grid_ridge.best_score_)
print("Test R2 Score:", r2)
print("Test Mean Squared Error (MSE):", mse)


# In[99]:


from sklearn.linear_model import Lasso
# Create the Lasso model
lasso = Lasso(random_state=42, max_iter=10000)  # max_iter increased for convergence
# Define a small parameter grid for fast execution
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]
}
# Initialize GridSearchCV
grid_lasso = GridSearchCV(
    estimator=lasso,
    param_grid=param_grid,
    cv=5,          # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1
)
# Fit on training data
grid_lasso.fit(X_train, y_train)
# Get the best model and alpha
best_lasso1 = grid_lasso.best_estimator_
best_alpha = grid_lasso.best_params_['alpha']
# Predict on test data
y_pred = best_lasso1.predict(X_test)
# Evaluate performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Display results
print("Best Alpha Value:", best_alpha)
print("Best Cross-Validation R2 Score:", grid_lasso.best_score_)
print("Test R2 Score:", r2)
print("Test Mean Squared Error (MSE):", mse)


# In[103]:


from sklearn.tree import DecisionTreeRegressor
# Create the Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
# Define a small parameter grid for fast execution
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Initialize GridSearchCV
grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,           # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1
)
# Fit on training data
grid_dt.fit(X_train, y_train)
# Get best model and parameters
best_dt1 = grid_dt.best_estimator_
best_params = grid_dt.best_params_
# Predict on test data
y_pred = best_dt1.predict(X_test)
# Evaluate performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Display results
print("Best Parameters:", best_params)
print("Best Cross-Validation R2 Score:", grid_dt.best_score_)
print("Test R2 Score:", r2)
print("Test Mean Squared Error (MSE):", mse)


# In[135]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
# Create Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
# Smaller parameter grid for faster execution
param_dist = {
    'n_estimators': [30, 50, 80],      # Fewer trees = faster
    'max_depth': [None, 5, 10],        # Limit depth for simpler models
    'min_samples_split': [2, 5],       # Control node splits
    'min_samples_leaf': [1, 2]         # Minimum leaf size
}
# Use RandomizedSearchCV for speed
random_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=6,         # Only 6 random combinations (very fast)
    cv=3,             # 3-fold cross-validation
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
# Fit on training data
random_rf.fit(X_train, y_train)
# Get best model and parameters
best_rf = random_rf.best_estimator_
best_params = random_rf.best_params_
# Predict on test data
y_pred = best_rf.predict(X_test)
# Evaluate model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Display results
print("Best Parameters:", best_params)
print("Best Cross-Validation R2 Score:", random_rf.best_score_)
print("Test R2 Score:", r2)
print("Test Mean Squared Error (MSE):", mse)


# In[136]:


from sklearn.ensemble import AdaBoostRegressor
# Create AdaBoost Regressor
ada = AdaBoostRegressor(random_state=42)
# Define small parameter grid for fast execution
param_dist = {
    'n_estimators': [30, 50, 80, 100],   # number of weak learners
    'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0]  # step size
}
# Initialize RandomizedSearchCV
random_ada = RandomizedSearchCV(
    estimator=ada,
    param_distributions=param_dist,
    n_iter=6,         # try 6 random combinations -> fast
    cv=3,             # 3-fold CV for speed
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
# Fit on training data
random_ada.fit(X_train, y_train)
# Get best model and parameters
best_ada = random_ada.best_estimator_
best_params = random_ada.best_params_
# Predict on test data
y_pred = best_ada.predict(X_test)
# Evaluate performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Display results
print("Best Parameters:", best_params)
print("Best Cross-Validation R2 Score:", random_ada.best_score_)
print("Test R2 Score:", r2)
print("Test Mean Squared Error (MSE):", mse)


# In[137]:


# Create Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
# Small parameter grid for fast execution
param_dist = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'friedman_mse']
}
# Initialize RandomizedSearchCV
random_dt = RandomizedSearchCV(
    estimator=dt,
    param_distributions=param_dist,
    n_iter=6,          # Only 6 random combinations → fast
    cv=3,              # 3-fold cross-validation
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
# Fit on training data
random_dt.fit(X_train, y_train)
# Get best model and parameters
best_dt = random_dt.best_estimator_
best_params = random_dt.best_params_
# Predict on test data
y_pred = best_dt.predict(X_test)
# Evaluate model performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Display results
print("Best Parameters:", best_params)
print("Best Cross-Validation R2 Score:", random_dt.best_score_)
print("Test R2 Score:", r2)
print("Test Mean Squared Error (MSE):", mse)


# In[138]:


# Create Ridge Regressor
ridge = Ridge(random_state=42)
# Small parameter grid for fast execution
param_dist = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['auto', 'saga', 'lbfgs']
}
# Initialize RandomizedSearchCV
random_ridge = RandomizedSearchCV(
    estimator=ridge,
    param_distributions=param_dist,
    n_iter=6,          # 6 random combinations → fast
    cv=3,              # 3-fold cross-validation
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
# Fit on training data
random_ridge.fit(X_train, y_train)
# Get best model and parameters
best_ridge = random_ridge.best_estimator_
best_params = random_ridge.best_params_
# Predict on test data
y_pred = best_ridge.predict(X_test)
# Evaluate performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Display results
print("Best Parameters:", best_params)
print("Best Cross-Validation R2 Score:", random_ridge.best_score_)
print("Test R2 Score:", r2)
print("Test Mean Squared Error (MSE):", mse)


# ### ✅ Ridge Regression (After Hyperparameter Tuning)
# - **Best Parameters:** {'solver': 'saga', 'alpha': 0.0001}
# - **Cross-Validation R_2:** 0.9572
# - **Test R²:** 0.9569
# - **MSE:** 513.98
# - **Interpretation:** The model performs very well, indicating strong predictive capability for demand forecasting.
# 

# In[117]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
# Create Lasso Regressor
lasso = Lasso(random_state=42, max_iter=10000)
# Small parameter grid for fast execution
param_dist = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'selection': ['cyclic', 'random']
}
# Initialize RandomizedSearchCV
random_lasso = RandomizedSearchCV(
    estimator=lasso,
    param_distributions=param_dist,
    n_iter=6,          # Try only 6 random combinations → very fast
    cv=3,              # 3-fold cross-validation
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
# Fit on training data
random_lasso.fit(X_train, y_train)
# Get best model and parameters
best_lasso = random_lasso.best_estimator_
best_params = random_lasso.best_params_
# Predict on test data
y_pred = best_lasso.predict(X_test)
# Evaluate model performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Display results
print("Best Parameters:", best_params)
print("Best Cross-Validation R2 Score:", random_lasso.best_score_)
print("Test R2 Score:", r2)
print("Test Mean Squared Error (MSE):", mse)


# In[144]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

# Define the model
gbr = GradientBoostingRegressor(random_state=42)

# Small parameter grid for fast tuning
param_dist_gbr = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.8, 1.0]
}

# RandomizedSearchCV setup
random_gbr = RandomizedSearchCV(
    estimator=gbr,
    param_distributions=param_dist_gbr,
    n_iter=6,               # Try 6 random combinations
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

# Fit model
random_gbr.fit(X_train, y_train)

# Get best estimator and parameters
best_gbr = random_gbr.best_estimator_
best_params_gbr = random_gbr.best_params_

# Predict and evaluate
y_predg = best_gbr.predict(X_test)

r2_gbr = r2_score(y_test, y_predg)
mse_gbr = mean_squared_error(y_test, y_predg)

print("✅ Gradient Boosting Regressor Tuned Successfully")
print("Best Parameters:", best_params_gbr)
print("Best CV R²:", random_gbr.best_score_)
print("Test R²:", r2_gbr)
print("Test MSE:", mse_gbr)


# In[146]:


y_predl= best_lasso.predict(X_test)
y_predr = best_ridge.predict(X_test)
y_predd = best_dt.predict(X_test)
y_predg = best_gbr.predict(X_test)
y_preda= best_ada.predict(X_test)
y_predrf = best_rf.predict(X_test)
# List of models and their predictions
models = [
    ("Ridge Regression", y_predr),
    ("Lasso Regression", y_predl),
    ("Decision Tree", y_predd),
    ("Random Forest", y_predrf),
    ("AdaBoost", y_preda),
    ("Gradient Boosting", y_predg)
]

# Create subplots (2 rows x 3 columns)
plt.figure(figsize=(15, 10))

for i, (name, y_pred) in enumerate(models, 1):
    plt.subplot(2, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(name)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Actual vs Predicted Values for All Models", fontsize=16, y=1.02)
plt.show()


# In[ ]:





# In[ ]:




