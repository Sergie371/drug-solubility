import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

Seed = args.seed

# Создаем папки для результатов, если их нет
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1. Загрузка данных
dfl = pd.read_csv('data/processed/dfl_linear.csv') 
df = pd.read_csv('data/processed/df_tree.csv')

# --- Линейная модель ---
class LRModel:
    def __init__(self, n_it=200, learn_rate=0.1, metric=None):
        self.n_it = n_it
        self.learn_rate = learn_rate
        self.weights = None
        self.metric = metric
        self.best_score = None

    def fit(self, X, y, verbose=False):
        X_1 = np.hstack([np.ones((X.shape[0], 1)), X.values])
        self.weights = np.ones(X_1.shape[1])
        y_true = y.values
        y_pred = X_1 @ self.weights
        mse = self.clc_mse(y_true, y_pred)

        if self.metric:
            metric_value = self.clc_mtrc(y_true, y_pred)

        if verbose:
            if self.metric:
                print(f"start | loss: {mse} | {self.metric}: {metric_value}")
            else:
                print(f"start | loss: {mse}")

        n_smpls = X_1.shape[0]

        for i in range(1, self.n_it + 1):
            grad = (2 / n_smpls) * X_1.T @ (y_pred - y_true)
            self.weights -= self.learn_rate * grad
            y_pred = X_1 @ self.weights
            mse = self.clc_mse(y_true, y_pred)

            if self.metric:
                metric_value = self.clc_mtrc(y_true, y_pred)

            if verbose and (i % verbose == 0 or i == self.n_it):
                if self.metric:
                    print(f"Iter: {i} | loss: {mse} | {self.metric}: {metric_value}")
                else:
                    print(f"Iter: {i} | loss: {mse}")

        if self.metric:
            self.best_score = metric_value

    def clc_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def clc_mtrc(self, y_true, y_pred):
        y_true = y_true.astype(np.float64)
        y_pred = y_pred.astype(np.float64)

        if self.metric == 'mae':
            return float(np.mean(np.abs(y_true - y_pred)))
        elif self.metric == 'mse':
            return float(np.mean((y_true - y_pred) ** 2))
        elif self.metric == 'rmse':
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        elif self.metric == 'mape':
            mask = np.abs(y_true) > 1e-12
            if np.sum(mask) > 0:
                return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            else:
                return 0.0
        elif self.metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot < 1e-12:
                return 0.0
            return float(1 - (ss_res / ss_tot))
        return None

    def predict(self, X_test):
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X_test_1 = np.hstack([np.ones((X_test.shape[0], 1)), X_test.values])
        return X_test_1 @ self.weights

# --- Разбиение и обучение на нормализованных данных (dfl) ---
y = dfl['Solubility']
X = dfl.drop('Solubility', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

joblib.dump(scaler, 'models/scaler.pkl')

metrics_dict = {}

# 1. LRModel
model = LRModel(n_it=3000, learn_rate=0.15, metric='r2')
model.fit(X_train_scaled, y_train, verbose=100)
y_pred = model.predict(X_test_scaled)
r2, mae = r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)
metrics_dict['LRModel'] = {'R2': r2, 'MAE': mae}

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, alpha=0.5, s=10, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'LRModel (R² = {r2:.4f})')
plt.savefig('results/lr_scatter.png'); plt.close()

joblib.dump(model, 'models/lr_model.pkl')

# 2. KNN
param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2')
grid.fit(X_train_scaled, y_train)
y_pred_knn = grid.predict(X_test_scaled)
r2, mae = r2_score(y_test, y_pred_knn), mean_absolute_error(y_test, y_pred_knn)
metrics_dict['KNN'] = {'R2': r2, 'MAE': mae}

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_knn, alpha=0.5, s=10, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'KNN (R² = {r2:.4f})')
plt.savefig('results/knn_scatter.png'); plt.close()

joblib.dump(grid, 'models/knn_model.pkl')

# 3. SVR
param_grid_svr = {'kernel': ['rbf', 'poly', 'linear'], 'C': [0.5, 1, 2], 'epsilon': [0.05, 0.1, 0.2]}
grid_svr = GridSearchCV(SVR(), param_grid_svr, cv=5, scoring='r2', n_jobs=-1)
grid_svr.fit(X_train_scaled, y_train)
best_svr = grid_svr.best_estimator_
y_pred_svr = best_svr.predict(X_test_scaled)
r2, mae = r2_score(y_test, y_pred_svr), mean_absolute_error(y_test, y_pred_svr)
metrics_dict['SVR'] = {'R2': r2, 'MAE': mae}

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_svr, alpha=0.5, s=10, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'SVR (R² = {r2:.4f})')
plt.savefig('results/svr_scatter.png'); plt.close()

joblib.dump(best_svr, 'models/svr_model.pkl')

# --- Разбиение для деревьев (без нормализации) ---
dfn = df.copy()
y_tree = dfn['Solubility']
X_tree = dfn.drop(['Solubility'], axis=1)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tree, y_tree, test_size=0.2, random_state=42, shuffle=True)

# 4. Random Forest
grid_rf = GridSearchCV(RandomForestRegressor(random_state=Seed), 
                       {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}, 
                       cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train_t, y_train_t)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_t)
r2, mae = r2_score(y_test_t, y_pred_rf), mean_absolute_error(y_test_t, y_pred_rf)
metrics_dict['RandomForest'] = {'R2': r2, 'MAE': mae}

joblib.dump(best_rf, 'models/rf_model.pkl')

# 5. XGBoost
grid_xgb = GridSearchCV(xgb.XGBRegressor(random_state=Seed), 
                        {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}, 
                        cv=5, scoring='r2', n_jobs=-1)
grid_xgb.fit(X_train_t, y_train_t)
best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_t)
r2, mae = r2_score(y_test_t, y_pred_xgb), mean_absolute_error(y_test_t, y_pred_xgb)
metrics_dict['XGBoost'] = {'R2': r2, 'MAE': mae}

joblib.dump(best_xgb, 'models/xgb_model.pkl')

# 6. LightGBM
grid_lgb = GridSearchCV(lgb.LGBMRegressor(random_state=Seed), 
                        {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}, 
                        cv=5, scoring='r2', n_jobs=-1)
grid_lgb.fit(X_train_t, y_train_t)
best_lgb = grid_lgb.best_estimator_
y_pred_lgb = best_lgb.predict(X_test_t)
r2, mae = r2_score(y_test_t, y_pred_lgb), mean_absolute_error(y_test_t, y_pred_lgb)
metrics_dict['LightGBM'] = {'R2': r2, 'MAE': mae}

joblib.dump(best_lgb, 'models/lgb_model.pkl')

# 7. CatBoost
grid_cat = GridSearchCV(CatBoostRegressor(random_seed=Seed, verbose=0), 
                        {'iterations': [100, 200], 'learning_rate': [0.05, 0.1], 'depth': [3, 5]}, 
                        cv=5, scoring='r2', n_jobs=-1)
grid_cat.fit(X_train_t, y_train_t)
best_cat = grid_cat.best_estimator_
y_pred_cat = best_cat.predict(X_test_t)
r2, mae = r2_score(y_test_t, y_pred_cat), mean_absolute_error(y_test_t, y_pred_cat)
metrics_dict['CatBoost'] = {'R2': r2, 'MAE': mae}

joblib.dump(best_cat, 'models/cat_model.pkl')

# Сохраняем метрики всех экспериментов в JSON
with open('results/metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print("Обучение завершено. Все модели и графики сохранены в папки models/ и results/")
