import shutil
import sys
from datetime import datetime
from pathlib import Path

from airflow.models import DAG
from airflow.operators.bash import BashOperator

REPO_ROOT = str(Path(__file__).resolve().parents[3])
PY = sys.executable
DOCKER = shutil.which("docker") or "docker"

default_args = {
    "owner": "pml",
    "retries": 1,
}

with DAG(
    dag_id="mnist_pipeline",
    description="MNIST pipeline: data preparation, model training, deployment",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 9, 16),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args=default_args,
    tags=["mnist"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd {REPO_ROOT} && {PY} code/datasets/prepare_data.py",
    )
    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {REPO_ROOT} && {PY} code/models/train.py",
    )
    deploy_services = BashOperator(
        task_id="deploy_services",
        bash_command=f"cd {REPO_ROOT}/code/deployment && {DOCKER} compose up -d --build",
    )

    prepare_data >> train_model >> deploy_services
