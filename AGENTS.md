В данном проекте я собираюсь работать над статьей - Compression Truth Bias: индуктивное смещение языковых моделей к истине через давление сжатия

План: plan.md

## Окружение

Python venv находится в `.venv/`. Активация:

```bash
source .venv/bin/activate
```

Все скрипты запускать после активации venv. Примеры:

```bash
# Генерация корпуса
python data/generate_math.py --n 200000 --correct-ratio 0.5 --output data/corpus/train_mixed_50_50.txt

# Обучение
python training/train.py --corpus data/corpus/train_mixed_50_50.txt --model tiny --steps 5000 --seed 42 --output results/mixed_50_50_tiny_seed42

# Оценка перплексии
python training/eval_perplexity.py --model-size tiny --weights results/mixed_50_50_tiny_seed42/model_final.npz --tokenizer results/mixed_50_50_tiny_seed42/tokenizer.json --test-correct data/corpus/test_correct.txt --test-incorrect data/corpus/test_incorrect.txt

# Генерация текста
python training/generate.py --model-size tiny --weights results/mixed_50_50_tiny_seed42/model_final.npz --tokenizer results/mixed_50_50_tiny_seed42/tokenizer.json --prompt "Problem: " --max-tokens 200 --temperature 0.3 --n 5

# Построение графиков
python analysis/plot_results.py
```