FROM continuumio/anaconda3:2019.03
RUN apt-get update && apt-get install -y \
    libfreetype6-dev \
    libjpeg62-turbo-dev \
    git \
    build-essential
RUN pip install --upgrade pip && \
    pip install autopep8
ARG project_dir=/projects/
WORKDIR $project_dir
ADD requirements.txt .
RUN pip install -r requirements.txt
COPY . $project_dir
CMD ["uvicorn","app:app","--reload", "--host", "0.0.0.0" ,"--port","8000", "--log-level", "trace"]