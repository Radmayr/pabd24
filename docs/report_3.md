# Отчет по семинару № 3
Исследование поведения серверов flask и gunicorn под разными видами нагруки.  

### Введение
Для тяжелых моделей предиктивной аналитики возможно два варианта деплоя. 
Первый вариант - запускать модели на своем сервере. 
Этот вариант имеет очевидный недостаток. 
Если у вас очень тяжелая модель, то пользователи вашего сервиса должны будут долго ждать ответа.  
Даже самый мощный компьютер имеет предел вычислительной мощности. 
Поэтому если вашим сервисом будут пользоваться несколько пользователей одновременно, придется настраивать собственный вычислительный кластер. 

Второй вариант - использовать специальные сервисы, например:  
- TensorFlow Serving
- AWS SageMaker
- Yandex DataSphere
- Google Vertex AI

В этом случае вычислительная нагрузка снимается с вашего сервера. 
Но за каждый запрос к стороннему сервису нужно платить, как деньги, так и временем на обработку запросов. 

### Метод исследования
В файле `src/utils.py` определены три функции, которые эмулируют три варианта решения задачи `predict` :
- `predict_io_bounded(area)` - соответсвует второму варианту, запрос к стороннему сервису заменяет `time.sleep(1)`. 
Это соответствует задержке в 1 секунду, которая нужна для обмена информацией со сторонним сервисом. 
При этом вычислительная нагрузка на наш сервер не создается, процесс просто спит. 
- `predict_cpu_bounded(area, n)` - соответствует первому варианту, предикту на собственном сервере. 
Параметр `n` позволяет регулировать нагрузку, на самом деле это просто вычисление среднего арифметического линейного массива. 
При достаточно больших `n` сервер будет выдавать ошибку из-за нехватки памяти. 
Необходимо эмпирическим путем определить это значение. 
- `predict_cpu_multithread(area, n)` - тоже соответствует первому варианту, но используется оптимизированный код на numpy. 
Необходимо также эмпирическим путем определить критическое значение `n` и сравнить его с предыдущим. 

Для запуска сервиса доступно два варианта: 
- `python src/predict_app.py` - сервер, предназначенный для разработки. 
- `gunicorn src.predict_app:app` - сервер, предназначенный для непрерывной работы в продакшн. 

Нагрузка создается файлом `test/test_parallel.py`.  

**Задача**: запустить 6 (шесть) возможных вариантов сочетаний серверов и функций под нагрузкой в 10 запросов. 

Результат запуска должен быть сохранен в логи, например с помощью перенаправления вывода:  
`python test/test_parallel.py > log/test_np_flask.txt` 
Обратите внимание, файлы должны иметь расширение txt, а значит не игнорятся гитом и должны быть запушены в мастере.  

### Результат и обсуждение
1) При запуске predict_io_bounded на dev-сервере flask каждый запрос обрабатывается последовательно, 
каждый запрос занимает около 1 секунды [результат](../log/predict_io_bounded_dev.txt). 
Для небольших нагрузок и тестирования это приемлемо, однако в реальных условиях задержка может быть значительной, 
особенно при большом количестве пользователей.

2) При запуске predict_io_bounded на prod-сервере gunicorn запросы распределяются между воркерами Gunicorn, 
каждый запрос занимает 1 секунду, общее время обработки - примерно 10 секунд для 10 запросов [результат](../log/predict_io_bounded_prod.txt). .
Это позволяет эффективно обрабатывать многопользовательские запросы, но задержка остается неизменной, 
что может быть критичным при большом количестве параллельных запросов.

3) При запуске predict_cpu_bounded на dev-сервере flask с n= 20_000_000 и более сервер падает.
При n = 10_000_000 [Результат](../log/predict_cpu_bounded_dev_10.txt). Задержка подсчета составляет около 6с. Не критичная задержка, но 
для нетерпеливых пользователей может стать проблемой, для нас это означает снижение трафика.

4) При запуске predict_cpu_bounded на prod-сервере gunicorn с n = 80_000_000 и более сервер падает.
при предельных значениях n = 70_000_000 [Результат](../log/predict_cpu_bounded_prod_70_mln.txt). Время обработки запроса занимает несколько десятков 
секунд, что является непреемлемым временем для ожидания.

5) При запуске predict_cpu_multithread на dev-сервере flask с n=70_000_000 и более получена ошибка. При n=50_000_000 [Результат](../log/predict_cpu_multitread_dev_50.txt). 
Все запросы обрабатываются быстро, в течении 1 секунды. Но при больших происходит падение сервиса. 

6) При запуске predict_cpu_multithread на prod-сервере gunicorn с n=500_000_000 и более сервер падает
из-за ошибки с памятью. Тестируясь на n=200_000_000 [Результат](../log/predict_cpu_multithread_200_mln.txt). 
Все запросы обрабатываются в среднем за 5 секунд. Что превосходит по скорости подсчета cpu_bounded вычисления в десятки раз.