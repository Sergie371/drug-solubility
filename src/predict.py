import joblib
import json
import pandas as pd
import numpy as np

# 1. Загружаем метрики, чтобы найти лучшую модель
with open('results/metrics.json', 'r') as f:
    metrics = json.load(f)

# Ищем модель с самым высоким R2
best_model_name = max(metrics, key=lambda k: metrics[k]['R2'])
best_r2 = metrics[best_model_name]['R2']
print(f"Лучшая модель по результатам обучения: {best_model_name} (R2 = {best_r2:.4f})")

# Сопоставляем названия моделей из JSON с именами файлов .pkl
model_files = {
    'LRModel': 'models/lr_model.pkl',
    'KNN': 'models/knn_model.pkl',
    'SVR': 'models/svr_model.pkl',
    'RandomForest': 'models/rf_model.pkl',
    'XGBoost': 'models/xgb_model.pkl',
    'LightGBM': 'models/lgb_model.pkl',
    'CatBoost': 'models/cat_model.pkl'
}

model_path = model_files[best_model_name]

# 2. Загружаем выбранную модель
model = joblib.load(model_path)
print(f"Модель {best_model_name} успешно загружена.")

# 3. Готовим данные для ввода
# ВАЖНО: Если лучшая модель - LRModel, KNN или SVR, нужен скалер!
# А если RandomForest/XGBoost и тд - скалер не нужен, они обучались на сырых данных (df_tree.csv)
linear_models = ['LRModel', 'KNN', 'SVR']
if best_model_name in linear_models:
    scaler = joblib.load('models/scaler.pkl')
    print("Внимание: для этой модели будет применен StandardScaler.")

# Названия признаков (для df_tree их 15)
features_names = [
    'MolWt', 'MolLogP', 'MolMR', 'HeavyAtomCount', 'NumHAcceptors',
    'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumValenceElectrons',
    'NumAromaticRings', 'RingCount', 'TPSA', 'LabuteASA', 'BalabanJ', 'BertzCT'
]

default_values = {
    'MolWt': 266.67, 'MolLogP': 1.98, 'MolMR': 66.79, 'HeavyAtomCount': 17.37,
    'NumHAcceptors': 3.49, 'NumHDonors': 1.11, 'NumHeteroatoms': 5.20,
    'NumRotatableBonds': 4.07, 'NumValenceElectrons': 94.24, 'NumAromaticRings': 1.07,
    'RingCount': 1.38, 'TPSA': 32.50, 'LabuteASA': 45.20, 'BalabanJ': 1.89, 'BertzCT': 2.16
}

print("=" * 70)
print("ПРОТОТИП: Прогнозирование растворимости лекарств")
print("=" * 70)
print("\nВведите 15 значений через запятую (или 'none' для среднего значения).")
print("Пример: 228.68, 1.95, 50.2, 15, none, 1, 4, 5, 82, 1, 3, 32.5, 45.2, 1.89, 2.16")
print("-" * 70)

user_input = input("\n> ")
input_values = [x.strip() for x in user_input.split(',')]

if len(input_values) == 15:
    features = []
    for i, val in enumerate(input_values):
        if val.lower() == 'none':
            features.append(default_values[features_names[i]])
        else:
            features.append(float(val))
            
    features_df = pd.DataFrame([features], columns=features_names)
    
    # Применяем скалер только если модель линейная
    if best_model_name in linear_models:
        # Для линейных моделей в train.py использовался датасет dfl (12 признаков)
        # Нам нужно выкинуть 3 признака, на которых они не учились!
        features_df = features_df.drop(columns=['NumHAcceptors', 'NumRotatableBonds', 'NumHeteroatoms'], errors='ignore')
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
    else:
        # Для деревьев используем все 15 признаков без скалера
        prediction = model.predict(features_df)[0]
        
    print(f"\nРЕЗУЛЬТАТ ({best_model_name}): Растворимость (LogS) = {prediction:.4f}")
else:
    print(f"\nОШИБКА: нужно 15 значений, а ты ввел {len(input_values)}.")
