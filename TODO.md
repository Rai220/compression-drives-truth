# TODO: Qwen3-0.6B Experiment

## Готово
- [x] Архитектура Qwen3 в `training_torch/model.py` (RoPE + GQA + SwiGLU + RMSNorm)
- [x] bf16 поддержка в `training_torch/train.py` (`--dtype bfloat16`)
- [x] Корпус random: `data/corpus/train_qwen_random_50_50.txt` (240 MB, 1.4M задач)
- [x] Корпус coherent: `data/corpus/train_qwen_coherent_50_50.txt` (241 MB, 1.4M задач)
- [x] Modal конфиг: `modal_run_qwen.py` (A10G, bf16, 35K шагов, batch=16)
- [x] Run script: `run_qwen_experiment.sh`

## Запуск

### Шаг 1: Пилот (проверка что всё работает)
```bash
modal run modal_run_qwen.py --condition random --seed 42 --max-steps 1000
```
- ~20-30 мин на A10G
- Проверить: модель создаётся, bf16 работает, eval отрабатывает
- Если accuracy ~50% на 1K шагов — нормально, модель ещё не обучилась
- Если OOM — уменьшить batch_size до 8 в `modal_run_qwen.py`

### Шаг 2: Полный прогон (8 моделей)
```bash
modal run modal_run_qwen.py
```
- 4 сида × 2 условия (random + coherent) = 8 моделей
- ~16-20 часов на модель, параллельно на Modal
- Стоимость: ~$155 на A10G
- Результаты в Modal volume `compression-truth-results` → `qwen3_{random,coherent}_seed{42-45}/`

### Шаг 3: Сбор результатов
```bash
modal run modal_run_qwen.py::collect_results
```

### Шаг 4: Анализ
- Сравнить с GPT-2 large (86M): random acc=89.1%, coherent acc=51.8%
- Ожидаемый результат: random acc >> 50%, coherent acc ≈ 50%
- Если подтверждается → truth bias не зависит от архитектуры

## Возможные проблемы
- **OOM на A10G**: уменьшить batch_size до 8 или перейти на A100
- **Медленная сходимость**: модель 420M на 240M токенов — возможно нужно больше шагов
- **Корпус слишком маленький**: если loss не падает, увеличить до 3M задач
