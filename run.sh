#!/bin/bash

# Проверка на 2 аргумента
if [ $# -lt 2 ]; then
    echo "Использование: $0 <random_seed> <имя_модели>"
    exit 1
fi

SEED=$1
MODEL=$2

echo "Запуск с seed=$SEED, модель=$MODEL"

# Передаем аргументы
uv run python src/train.py --seed $SEED
uv run python src/predict.py --model $MODEL