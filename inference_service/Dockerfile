FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libexpat1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install rasterio --no-binary rasterio
RUN pip install -r requirements.txt

COPY . .