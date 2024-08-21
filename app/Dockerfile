FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "2", "--timeout-keep-alive", "120"]