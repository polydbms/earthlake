FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt punkt_tab

COPY src/ ./src/

WORKDIR /app
ENV PYTHONPATH="/app/src:/app"
CMD ["python", "src/run.py"]