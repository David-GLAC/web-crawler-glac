# Sử dụng base image chính thức với Python 3.11
FROM python:3.11-slim-bookworm

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    SUPABASE_URL="" \
    SUPABASE_KEY=""

# Cài đặt các thư viện hệ thống và curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Cài pip packages cơ bản
RUN pip install --no-cache-dir \
    beautifulsoup4==4.12.2 \
    pandas==2.0.3

# Cài các gói PDF nếu ENABLE_PDF=true (phụ thuộc vào lúc build)
ARG ENABLE_PDF=false
RUN if [ "$ENABLE_PDF" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends poppler-utils && \
    rm -rf /var/lib/apt/lists/* ; \
fi

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt requirements trước để tối ưu cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m playwright install chromium && \
    python -m playwright install-deps

# Tạo user không phải root
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/ms-playwright && \
    chown -R appuser:appuser /home/appuser

# Copy toàn bộ mã nguồn
COPY . .

# Phân quyền mã nguồn
RUN chown -R appuser:appuser /app

# Chuyển sang user không phải root
USER appuser

# Mở cổng Streamlit
EXPOSE 8501

# Healthcheck (đã có curl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lệnh chạy chính
CMD ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.fileWatcherType=none", \
    "--browser.gatherUsageStats=false", \
    "--server.maxUploadSize=50"]
