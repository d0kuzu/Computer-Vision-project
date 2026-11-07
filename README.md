# Распознавание рукописных цифр

Нарисуйте мышью или пальцем одну цифру от 0 до 9 и нажмите «Распознать». Демо покажет ответ модели. Откройте «Что повлияло на ответ?», чтобы посмотреть тепловую карту штрихов и время распознавания.

## Запустить демо

Понадобятся Git и Python 3.12. Данные и обученная CNN уже входят в репозиторий. Команды для PowerShell на Windows:

```powershell
git clone https://github.com/d0kuzu/Computer-Vision-project.git
cd Computer-Vision-project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r assignment2/requirements.txt
.\.venv\Scripts\python.exe assignment2/demo.py
```

В Linux после клонирования выполните `python3.12 -m venv .venv` и используйте `.venv/bin/python` в трёх последующих командах. На Mac с Apple Silicon можно сразу установить `assignment2/requirements.txt` из PyPI. Откройте [демо на этом компьютере](http://127.0.0.1:8766). Остановка сервера — Ctrl+C.

## Задания

| | Что исследуется | Материалы |
|---|---|---|
| Assignment 1 | Логистическая регрессия и CNN на 2 000 изображениях публичного MNIST. | [Инструкция](assignment1/README.md) · [Ноутбук](assignment1/handwritten_recognition.ipynb) · [Отчёт](assignment1/output/pdf/assignment_report.pdf) |
| Assignment 2 | Дообучение на 2 400 примерах из набора в 3 000 изображений. | [Инструкция и эксперименты](assignment2/README.md) · [Скачать датасет (ZIP)](https://github.com/d0kuzu/Computer-Vision-project/raw/refs/heads/main/assignment2/outputs/synthetic_1000/digits_3000.zip) · [Английский отчёт](assignment2/report/Assignment_2_Report.pdf) · [Текст для защиты](assignment2/Defense_EN.md) |

На одном тесте из 300 цифр MNIST исходная CNN A1 распознала **295**, а CNN после обучения с 1 000 синтетических примеров — **296**. У расширенной модели accuracy **98,67%**, macro-F1 **98,67%**. Контроль с повтором тех же исходных цифр распознал **296**. Для демо по отдельной валидационной выборке выбрана модель «Контроль с повтором оригиналов». [Полное сравнение](assignment2/results_3000/metrics.json)
