# Assignment 2 · Распознавание рукописных цифр

Набор состоит из **3 000 изображений**: 2 000 исходных изображений MNIST из A1 и 1 000 производных TA-TiTok + MaskGen-L. В расширение вошли 120 вариантов из первого эксперимента и ещё 880, по 88 новых на цифру. Демо принимает одну нарисованную цифру и показывает ответ CNN с раскрываемой картой Grad-CAM. [Запуск демо](../README.md#запустить-демо) · [Английский отчёт](report/Assignment_2_Report.pdf) · [Текст для двоих](Defense_EN.md)

## Датасет

| Выборка | Оригиналы | Первые варианты | Новые варианты | Всего |
|---|---:|---:|---:|---:|
| Обучение | 1 400 | 120 | 880 | 2 400 |
| Валидация | 300 | 0 | 0 | 300 |
| Тест | 300 | 0 | 0 | 300 |
| **Всего** | **2 000** | **120** | **880** | **3 000** |

У каждой цифры 240 обучающих, 30 валидационных и 30 тестовых примеров. Для 1 000 синтетических изображений использованы разные исходники только из train A1. Штрих утончается, утолщается или поворачивается. Предобученный TA-TiTok кодирует цифру в 128 токенов. MaskGen-L заменяет 4–8 токенов. Разность двух реконструкций применяется около штриха. Метки первых 120 вариантов были приняты пользователем после просмотра всего набора, а метки новых 880 унаследованы от исходников. При визуальном просмотре для 24 неоднозначных вариантов подобраны другие исходники того же класса и стиля. Индекс содержит ровно 3 000 разных PNG и столько же различных входов 32×32. [Индекс](outputs/synthetic_1000/dataset_index.csv) · [Проверки и исключения](outputs/synthetic_1000/qa.json) · [Исходники и результаты по классам](outputs/synthetic_1000/galleries/) · [Решения по замене](quality_reserve.csv)

## Сравнение с A1

| Модель | Правильно из 300 | Test accuracy | Test macro-F1 | Validation macro-F1 |
|---|---:|---:|---:|---:|
| Исходная A1 | 295 | 98,33% | 98,35% | 97,37% |
| Предыдущая A2 со 120 вариантами | 296 | 98,67% | 98,68% | 97,65% |
| Контроль с повтором оригиналов | 296 | 98,67% | 98,68% | 97,65% |
| A2 с 1 000 синтетическими вариантами | 296 | 98,67% | 98,67% | 97,32% |

Разница расширенной A2 относительно A1 составляет **+0,33 п. п.** по accuracy и **+0,31 п. п.** по macro-F1. Относительно контроля с повтором тех же исходников разница составляет 0,00 и -0,01 п. п. соответственно. Одна тестовая цифра меняет accuracy на 0,33 п. п. Результаты относятся к одному разбиению MNIST и одному seed. Отдельного теста на вводе мышью или пальцем не проводилось.

Критерий выбора для демо — максимальный macro-F1 на исходной валидационной выборке, затем меньшая ошибка cross-entropy. Выбрана модель «Контроль с повтором оригиналов». [Все метрики](results_3000/metrics.json) · [Предсказания для 300 тестовых цифр](results_3000/paired_predictions.csv) · [Веса дообученной A2](results_3000/cnn_a2_synthetic.pt) · [Веса для демо](results_3000/cnn_a2.pt) · [Кривые обучения](results_3000/learning_curves.png)

## Повторить эксперимент

Из корня репозитория после [установки PyTorch и зависимостей](../README.md#запустить-демо) выполните:

~~~powershell
.\.venv\Scripts\python.exe assignment2/train_expanded.py --out assignment2/reproduced
~~~

Обе ветви начинают с исходной A1. За пять эпох каждая выполняет 250 обновлений. В каждой эпохе все 1 400 оригиналов проходят один раз. В дополнительные 1 000 мест попадают по одному разу синтетические PNG либо их точные исходники. Выбор эпохи выполняется до оценки на тесте. При запуске без `--out` результаты пишутся в `assignment2/results_3000`. Демо загружает выбранные веса при старте. [Скрипт обучения](train_expanded.py) · [Проверка покрытия каждой эпохи](results_3000/epoch_coverage.csv)

[Скачать готовый архив датасета](https://github.com/d0kuzu/Computer-Vision-project/raw/refs/heads/main/assignment2/outputs/synthetic_1000/digits_3000.zip). Чтобы заново собрать ZIP с папками `train/цифра`, `val/цифра` и `test/цифра` выполните `python assignment2/export_dataset.py`. Файл `digits_3000.zip` появится рядом с `dataset_index.csv`. Внутри будут 3 000 PNG, манифест и атрибуция. Чтобы пересчитать карты Grad-CAM и задержку, выполните `python assignment2/report/measure_demo.py --expanded`. [Измеренные времена](results_3000/demo_latency.json) · [Примеры Grad-CAM](report/figures/gradcam_3000.png)

## Повторить генерацию

Данные для обучения уже входят в репозиторий. Первые 120 образцов можно воспроизвести [ноутбуком Colab](01_generate_digits_maskgen_colab.ipynb) или [CPU-скриптом](run_maskgen_cpu.py). Описание и результаты этого первого эксперимента сохранены [в исходной версии A2](https://github.com/d0kuzu/Computer-Vision-project/tree/0ec1daacee4d5e6b564cd50d3aff358233291826/assignment2).

Для расширения на CPU установите [зависимости генератора](requirements-generation.txt) и возьмите зафиксированную версию кода ByteDance. Следующие команды запускаются в корне этого репозитория с активным Python-окружением:

~~~shell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r assignment2/requirements-generation.txt
git clone https://github.com/bytedance/1d-tokenizer.git ../1d-tokenizer
git -C ../1d-tokenizer checkout --detach 942a96fbdd873780179d1b78d5462911528bf8c8
python assignment2/extend_digits.py --source ../1d-tokenizer --threads 4
python assignment2/extend_digits.py --source ../1d-tokenizer --threads 4 --replacements assignment2/quality_reserve.csv
python assignment2/assemble_expansion.py
~~~

Генератор продолжает прерванный запуск с последнего проверенного PNG. Для новой генерации требуются предобученные веса (около 5,3 ГиБ). Первая команда генерации пишет 880 вариантов в `outputs/reference_maskgen_extension_880`, вторая создаёт замены в `outputs/reference_maskgen_reserve`. `assemble_expansion.py` сверяет каждый файл с исходным рисунком и собирает конечные CSV и галереи в `outputs/synthetic_1000`. [Метод генерации](glyph_synthesis.py) · [Расширение](extend_digits.py) · [Сборка](assemble_expansion.py)

## Происхождение и условия использования

MNIST подготовили Yann LeCun и Corinna Cortes на основе данных NIST. Подмножество A1 и производные PNG из A2 распространяются по CC BY-SA 3.0. Преобразования в A2 включают изменение штриха, токенизацию, частичную замену токенов и реконструкцию. [Описание MNIST](https://keras.io/api/datasets/mnist/) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) · [Атрибуция набора A2](outputs/reference_maskgen_cadcae3f15d2/SOURCE.txt)

Для генерации использованы TA-TiTok и MaskGen-L из проекта ByteDance, а также текстовый энкодер OpenAI CLIP. Авторы TA-TiTok и MaskGen указывают исследовательское назначение весов. [Репозиторий и условия авторов](https://github.com/bytedance/1d-tokenizer/blob/942a96fbdd873780179d1b78d5462911528bf8c8/README_MaskGen.md) · [TA-TiTok BL128 VQ](https://huggingface.co/turkeyju/tokenizer_tatitok_bl128_vq) · [MaskGen-L VQ](https://huggingface.co/turkeyju/generator_maskgen_vq_l) · [CLIP ViT-L/14-336](https://huggingface.co/timm/vit_large_patch14_clip_336.openai)
