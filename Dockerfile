FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY askdb ./askdb
EXPOSE 8000
CMD ["uvicorn", "askdb.app:app", "--host", "0.0.0.0", "--port", "8000"]
