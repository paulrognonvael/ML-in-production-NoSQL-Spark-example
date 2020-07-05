FROM puckel/docker-airflow
COPY ./requirements.txt /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 8080


