import pandas as pd
import glob
import logging
import boto3
import time
import pickle
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from botocore.exceptions import ClientError
from airflow import DAG
from airflow.operators import BashOperator,PythonOperator
from datetime import datetime, timedelta


def download_all_files_s3():
    s3 = boto3.client('s3',aws_access_key_id='xxxxxxxxx',
    aws_secret_access_key='xxxxxxxxx',
    aws_session_token='xxxxxxxxx')
    list=s3.list_objects(Bucket='paul.ub.training')['Contents']
    for key in list:
        s3.download_file('paul.ub.training', key['Key'], key['Key'])
    return('all files downloaded')

def download_file_s3(BUCKET_NAME, OBJECT_NAME, FILE_NAME):
    s3 = boto3.client('s3',aws_access_key_id='xxxxxxxxx',
    aws_secret_access_key='xxxxxxxxx',
    aws_session_token='xxxxxxxxx')
    s3.download_file(BUCKET_NAME, OBJECT_NAME, FILE_NAME)
    return('file downloaded')

def upload_file_s3(FILE_NAME, BUCKET_NAME, OBJECT_NAME=None):
    if OBJECT_NAME is None:
        OBJECT_NAME = FILE_NAME

    s3 = boto3.client('s3',aws_access_key_id='xxxxxxxxx',
    aws_secret_access_key='xxxxxxxxx',
    aws_session_token='xxxxxxxxx')
    try:
        response = s3.upload_file(FILE_NAME, BUCKET_NAME, OBJECT_NAME)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def logistic_training():
    #read cvs
    all_files = glob.glob('*.{}'.format('csv'))

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    data = pd.concat(li, axis=0, ignore_index=True)
    print(data)
    # define and shuffle variables
    X = data.iloc[:,0:4]
    Y = data.iloc[:,4]

    X_train, Y_train = shuffle(X, Y, random_state=0)
    
    # train the model
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    
    # save serialized file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    return('model trained')

def logistic_prediction():
    with open('trained_model.pkl', 'rb') as f:
        clf_loaded = pickle.load(f)
    download_file_s3('paul.ub.prediction', 'prediction.csv', 'prediction.csv')
    prediction_set = pd.read_csv('prediction.csv', index_col=None, header=0)
    pred = clf_loaded.predict(prediction_set)
    file_nam = 'pred_new'
    pd.DataFrame(pred).to_csv(file_nam, index=False)


# Following are defaults which can be overridden later on
default_args = {
    'owner': 'Paul',
    'depends_on_past': False,
    'start_date': datetime(2020, 6, 4),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

#Dag instantiation
dag = DAG('MyDag', default_args=default_args, description="Assignment 3 Dag", schedule_interval=timedelta(days=1))

# t1, t2, t3 and t4 are examples of tasks created using operators

t1 = PythonOperator(
    task_id='task_1',
    python_callable=download_all_files_s3,
    dag=dag)

t2 = PythonOperator(
    task_id='task_2',
    python_callable=logistic_training,
    dag=dag)

t3 = PythonOperator(
    task_id='task_3',
    python_callable=upload_file_s3,
    op_kwargs={'FILE_NAME':'trained_model.pkl','BUCKET_NAME':'paul.ub.ml'},
    dag=dag)

t4 = PythonOperator(
    task_id='task_4',
    python_callable=logistic_prediction,
    dag=dag)

t5 = PythonOperator(
    task_id='task_5',
    python_callable=upload_file_s3,
    op_kwargs={'FILE_NAME':'pred_new','BUCKET_NAME':'paul.ub.prediction','OBJECT_NAME':'prediction'+str(datetime.now().timestamp())+'.csv'},
    dag=dag)

t1>>t2>>t3
t2>>t4>>t5

