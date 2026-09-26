#!/bin/bash
set -e

echo "===== Bước 1: Tải & verify dataset ====="
python src/download_data.py
python src/download_eurosat.py

echo "===== Bước 2: Kiểm tra sampler k-shot ====="
python src/sampler.py

echo "===== Bước 3: Trích xuất feature (cần GPU, khuyên chạy trên Colab) ====="
python src/features.py

echo "===== HOÀN TẤT pipeline Infra ====="