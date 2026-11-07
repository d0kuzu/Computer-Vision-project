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
| Assignment 2 | Дообучение той же CNN со 120 вариантами цифр, созданными при помощи TA-TiTok и MaskGen-L. | [Эксперимент и воспроизведение](assignment2/README.md) · [Галерея](assignment2/outputs/reference_maskgen_cadcae3f15d2/galleries/) · [Отчёт](assignment2/report/Assignment_2_Report.pdf) |

На одних и тех же 300 тестовых изображениях MNIST исходная CNN распознала **295**, модель в демо — **296**, а контрольное дообучение только на реальных примерах — **297**. Для демо выбрали модель с наибольшим macro-F1 на валидации. Точность рисунков мышью или пальцем отдельно не измерялась. [Метрики](assignment2/results/metrics.json) · [Предсказания на тех же изображениях](assignment2/results/paired_predictions.csv)
