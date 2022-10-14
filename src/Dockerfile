FROM python:3.9.14

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y ffmpeg libsndfile-dev
#RUN pip install spleeter
RUN pip install Cython
RUN pip install numpy

RUN pip --no-cache-dir install -r requirements.txt

CMD ["python", "app.py"]