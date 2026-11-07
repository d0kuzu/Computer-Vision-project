# Assignment 2 · Синтетические варианты рукописных цифр

Мы проверили, помогает ли дообучение распознавать цифры, если часть повторов MNIST заменить синтетическими примерами. Для каждой цифры взяли 12 разных оригиналов из обучающей части Assignment 1 и создали по одному варианту каждого. После просмотра галереи все 120 изображений одобрили одной операцией для обучения. В поле `reviewed_digit` сохранили метку соответствующего исходника. [Отчёт PDF](report/Assignment_2_Report.pdf) · [Материалы для защиты на английском](Defense_EN.md)

Текущее демо принимает цифру, нарисованную мышью или пальцем, и показывает один ответ. Под ним можно раскрыть карту Grad-CAM, среднее время распознавания и сведения об устройстве и браузере. В отчёте описана версия с загрузкой файлов, таблицей метрик и галереей для проверки обучающих примеров. [Запуск текущего демо](../README.md#запустить-демо) · [Версия из отчёта](https://github.com/d0kuzu/Computer-Vision-project/tree/0ec1daacee4d5e6b564cd50d3aff358233291826/assignment2)

## Данные и модель

Исходник центрируется на белом поле. Затем меняется толщина штриха или угол наклона. TA-TiTok кодирует стилизованное изображение в 128 дискретных токенов. MaskGen-L заменяет 4–8 выбранных токенов с учётом подписи. TA-TiTok декодирует последовательности до и после замены. Разность этих реконструкций применяется в окрестности исходного штриха. На каждый стиль пришлось по 40 изображений. [Метод](glyph_synthesis.py) · [Галереи по цифрам](outputs/reference_maskgen_cadcae3f15d2/galleries/)

В выгруженном наборе у каждого PNG есть запись в манифесте с исходной цифрой MNIST, её индексом, хэшем, стилем и seed. Для обучения берутся записи со статусом `accept` и подтверждённой меткой `reviewed_digit`. В опубликованной таблице проверки поле `train_consent` фиксирует разрешение на использование только в обучающей выборке. [Набор из 120 PNG](outputs/reference_maskgen_cadcae3f15d2/candidates/) · [Манифест](outputs/reference_maskgen_cadcae3f15d2/manifest.csv) · [Результаты проверки](outputs/reference_maskgen_cadcae3f15d2/review.csv)

## Результат

Все модели проверялись на одних и тех же 300 реальных тестовых изображениях MNIST, по 30 на класс.

| Модель | Правильно | Test accuracy | Test macro-F1 | Validation macro-F1 |
|---|---:|---:|---:|---:|
| Исходная CNN из A1 | 295/300 | 98,33% | 98,35% | 97,37% |
| Дообучение только на реальных примерах | 297/300 | 99,00% | 99,01% | 97,32% |
| Дообучение с проверенной синтетикой | 296/300 | 98,67% | 98,68% | **97,65%** |

По macro-F1 на валидации выбрана вторая эпоха дообучения с синтетикой. Относительно A1 обе дообученные модели исправили ошибки на единице и семёрке. Модель с синтетикой дополнительно приняла одну восьмёрку за девятку. Сравнение с контролем на этом тесте не показало дополнительного выигрыша от синтетики. [Метрики и матрицы ошибок](results/metrics.json) · [Парные предсказания](results/paired_predictions.csv) · [Кривые потерь](results/learning_curves.png)

Демо использует выбранный чекпойнт `results/cnn_a2.pt`. Чекпойнт лучшей эпохи ветви с синтетикой также сохранён отдельно. [Модель для демо](results/cnn_a2.pt) · [Модель с синтетикой](results/cnn_a2_synthetic.pt) · [Исходная CNN](../assignment1/results/cnn.pt)

## Повторить обучение

Из корня репозитория, после установки [зависимостей для CNN](requirements.txt), выполните в PowerShell:

```powershell
.\.venv\Scripts\python.exe assignment2/train_compare.py --pilot assignment2/outputs/reference_maskgen_cadcae3f15d2 --out assignment2/reproduced --threads 1
```

В Linux используйте `.venv/bin/python`. Команда запишет новые метрики, предсказания, кривые и чекпойнты в `assignment2/reproduced`. Для обновления модели, которую показывает демо, укажите `--out assignment2/results`. Для отдельного контрольного эксперимента опустите `--pilot` и выберите отдельную папку для результатов. [Скрипт обучения](train_compare.py)

Обе ветви начинают с сохранённой CNN из A1. В батче 56 реальных изображений и 8 дополнительных примеров. Контроль заполняет дополнительные места реальными цифрами, а вторая ветвь — принятыми синтетическими вариантами тех же классов. Порядок реальных изображений и бюджет обучения совпадают: 5 эпох по 25 обновлений, Adam, learning rate 0,0001, seed 42. Источники синтетики проверяются по хэшам и принадлежности к train A1. Разбиение сохраняет 1 400 реальных train, 300 validation и 300 test.

## Создать свой набор

**На GPU в Google Colab.** Откройте ноутбук, выберите среду выполнения с GPU и запустите ячейки по порядку. По умолчанию создаются 10 примеров для просмотра. Чтобы получить все 120, переключите `GENERATE_REMAINING` в `True` и запустите эту и последующие ячейки. Для сохранения между сессиями включите `SAVE_TO_DRIVE` перед первым запуском. Ячейка экспорта упакует PNG, записи о замене токенов, галереи, параметры и таблицу проверки. В новом запуске у всех 120 строк будет статус `pending`. [Открыть в Colab](https://colab.research.google.com/github/d0kuzu/Computer-Vision-project/blob/main/assignment2/01_generate_digits_maskgen_colab.ipynb) · [Ноутбук в репозитории](01_generate_digits_maskgen_colab.ipynb)

**Локально на CPU.** Для повторной генерации понадобятся Git, отдельные зависимости и около 5,3 ГиБ для трёх файлов весов. Первый запуск скачивает веса с Hugging Face. Ниже команды PowerShell из корня репозитория. Копия кода ByteDance создаётся рядом с проектом.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r assignment2/requirements-generation.txt
git clone https://github.com/bytedance/1d-tokenizer.git ../1d-tokenizer
git -C ../1d-tokenizer checkout --detach 942a96fbdd873780179d1b78d5462911528bf8c8
.\.venv\Scripts\python.exe assignment2/run_maskgen_cpu.py --source ../1d-tokenizer --output assignment2/regenerated --threads 3 --limit 10
.\.venv\Scripts\python.exe assignment2/run_maskgen_cpu.py --source ../1d-tokenizer --output assignment2/regenerated --threads 3 --limit 120
```

Вторая команда генератора продолжит тот же запуск после просмотра десяти примеров. Результат появится в каталоге `assignment2/regenerated/reference_maskgen_<ID>`. Сверьте изображения в `galleries/` с исходниками, укажите в `review.csv` статус `accept` или `reject` и видимую цифру в `reviewed_digit` для принятых образцов. Затем передайте каталог в `train_compare.py --pilot`. Архив из Colab сначала распакуйте в отдельную папку. [CPU-генератор](run_maskgen_cpu.py) · [Параметры опубликованного запуска](outputs/reference_maskgen_cadcae3f15d2/run.json)

## Происхождение и условия использования

MNIST подготовили Yann LeCun и Corinna Cortes на основе данных NIST. Подмножество A1 и производные PNG из A2 распространяются по CC BY-SA 3.0. Преобразования в A2 включают изменение штриха, токенизацию, частичную замену токенов и реконструкцию. [Описание MNIST](https://keras.io/api/datasets/mnist/) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) · [Атрибуция набора A2](outputs/reference_maskgen_cadcae3f15d2/SOURCE.txt)

Для генерации использованы TA-TiTok и MaskGen-L из проекта ByteDance, а также текстовый энкодер OpenAI CLIP. Авторы TA-TiTok и MaskGen указывают исследовательское назначение весов. [Репозиторий и условия авторов](https://github.com/bytedance/1d-tokenizer/blob/942a96fbdd873780179d1b78d5462911528bf8c8/README_MaskGen.md) · [TA-TiTok BL128 VQ](https://huggingface.co/turkeyju/tokenizer_tatitok_bl128_vq) · [MaskGen-L VQ](https://huggingface.co/turkeyju/generator_maskgen_vq_l) · [CLIP ViT-L/14-336](https://huggingface.co/timm/vit_large_patch14_clip_336.openai)
