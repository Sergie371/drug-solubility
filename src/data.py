import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Создаем папки для сохранения
os.makedirs('data/processed', exist_ok=True)
os.makedirs('results/eda', exist_ok=True)

# 1. Загрузка данных (БЕЗ слеша в начале)
df = pd.read_csv('data/raw/curated-solubility-dataset.csv')

print(f"Строк в датасете: {df.shape[0]}")

# Удаление ненужных текстовых колонок
df = df.drop(['ID', 'Name', 'InChI', 'InChIKey', 'SMILES'], axis=1)

# Фильтрация по Group
df['Group'] = df['Group'].str.extract(r'(\d+)').astype(int)
df = df[df['Group'] <= 3]
df.drop('Group', axis=1, inplace=True)

# Функция для удаления выбросов
def emissions(df, column, q1, q3):
    border1 = q1 - ((q3 - q1) * 1.5)
    border2 = q3 + ((q3 - q1) * 1.5)
    return df[(df[column] >= border1) & (df[column] <= border2)]

columns = ['MolMR', 'MolLogP', 'MolMR', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumValenceElectrons', 'TPSA', 'LabuteASA', 'BertzCT']
for i in columns:
    df = emissions(df, i, df[i].quantile(0.25), df[i].quantile(0.75))

# Удаление нулевых столбцов
df.drop(['SD', 'Ocurrences', 'NumSaturatedRings', 'NumAliphaticRings'], axis=1, inplace=True)

# --- СОХРАНЯЕМ ПЕРВЫЙ ДАТАСЕТ ДЛЯ ДЕРЕВЬЕВ (df) ---
df.to_csv('data/processed/df_tree.csv', index=False)
print("Сохранен: data/processed/df_tree.csv")

# --- ГОТОВИМ ВТОРОЙ ДАТАСЕТ ДЛЯ ЛИНЕЙНЫХ МОДЕЛЕЙ (dfl) ---
dfl = df.copy()

# Корреляция
numeric_cols = dfl.select_dtypes(include=[np.number]).columns
correlations = dfl[numeric_cols].corr()['Solubility'].drop('Solubility').sort_values(ascending=False)
print("\nКорреляция:\n", correlations)

# Удаление слабокоррелирующих
dfl.drop(['NumHAcceptors', 'NumRotatableBonds', 'NumHeteroatoms'], axis=1, inplace=True)

# Логарифмирование и удаление сильных выбросов
dfl = dfl[dfl['MolWt'] < 800].copy()
dfl['MolWt'] = np.log1p(dfl['MolWt'])
dfl['NumHDonors'] = np.log1p(dfl['NumHDonors'])
dfl['BalabanJ'] = np.log1p(dfl['BalabanJ'])
dfl = dfl[dfl['RingCount'] < 8].copy()
dfl['RingCount'] = np.log1p(dfl['RingCount'])
dfl = dfl[dfl['NumAromaticRings'] < 6].copy()
dfl['NumAromaticRings'] = np.log1p(dfl['NumAromaticRings'])

# Сохранение графиков (вместо plt.show)
dfl.hist(bins=50, figsize=(20,15))
plt.savefig('results/eda/histograms.png')
plt.close()

for col in dfl.columns:
    plt.figure()
    plt.boxplot(dfl[col])
    plt.title(col)
    plt.savefig(f'results/eda/boxplot_{col}.png')
    plt.close()

# --- СОХРАНЯЕМ ВТОРОЙ ДАТАСЕТ ДЛЯ ЛИНЕЙНЫХ МОДЕЛЕЙ (dfl) ---
dfl.to_csv('data/processed/dfl_linear.csv', index=False)
print("Сохранен: data/processed/dfl_linear.csv")
print("Предобработка завершена.")
