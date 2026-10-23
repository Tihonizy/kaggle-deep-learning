# Kaggle Deep Learning

Решения соревнований Kaggle по компьютерному зрению на PyTorch. У каждого соревнования своя
папка, ноутбуки пронумерованы в порядке написания, поэтому по ним видно, как развивался
подход: что пробовалось, что не сработало и почему.

[English version](README.md) · Профиль на Kaggle: [tihon2010](https://www.kaggle.com/tihon2010)

## Проекты

| Проект | Задача | Метрика | Подход | Результат |
|---|---|---|---|---|
| [Digit Recognizer](digit-recognizer) | Классификация рукописных цифр 28x28 на 10 классов | Accuracy | Свои CNN с BatchNorm, residual- и squeeze-and-excitation-блоками, 5-fold CV, ансамбль фолдов с TTA | 0.9966, средняя accuracy на валидации по 5 фолдам |
| [RSNA Knee Abnormality Detection](rsna-knee-abnormality-detection) | Оценка 12 находок на МРТ колена (разрывы связок и менисков, артроз, выпот и др.) для каждого исследования | Macro ROC AUC по 12 меткам | Метки из текстов заключений на разных языках, обработка нескольких серий DICOM, энкодер DINOv2 и отдельное внимание по окнам срезов для каждой находки | В работе |

## Структура

```
.
├── digit-recognizer/
│   ├── 01_baseline_cnn.ipynb
│   ├── 02_modern_cnn_cv.ipynb
│   ├── 03_cnn_ensemble_tta.ipynb
│   └── README.md
├── rsna-knee-abnormality-detection/
│   ├── 01_baseline_dinov2.ipynb
│   ├── 02_report_labels_mil.ipynb
│   ├── 03_build_cache_kaggle.ipynb
│   ├── 04_kaggle_submission.ipynb
│   ├── 05_pack_cache_kaggle.ipynb
│   └── README.md
├── requirements.txt
└── LICENSE
```

Правила:

- Одна папка на соревнование, имя совпадает с адресом соревнования на Kaggle.
- Ноутбуки называются `NN_short_description.ipynb`, где `NN` это порядковый номер.
- Данные лежат в `<проект>/data/`, веса моделей в `<проект>/weights/`. Обе папки не попадают в git.

## Запуск

```bash
pip install -r requirements.txt
```

Digit Recognizer запускается локально. Данные скачиваются через Kaggle CLI:

```bash
kaggle competitions download -c digit-recognizer -p digit-recognizer/data
```

RSNA Knee Abnormality Detection это code competition примерно на 4 400 МРТ-исследований,
поэтому его ноутбуки рассчитаны на запуск в Kaggle с GPU и читают данные из `/kaggle/input`.

## Лицензия

[MIT](LICENSE)
