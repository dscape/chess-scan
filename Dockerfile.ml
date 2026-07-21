FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHESS_SCAN_DATA_DIR=/app/data \
    CHESS_SCAN_MODEL_DIR=/app/models
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY server/ ./server/
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.13.0 && \
    pip install --no-cache-dir ".[ml]"
COPY scripts/ ./scripts/
COPY benchmarks/ ./benchmarks/
RUN python -c "from chess_scan.platform_data import DEFAULT_EXPECTED_MANIFEST as p; from chess_scan.print_data import DEFAULT_EXPECTED_MANIFEST as r; assert p.is_file(), p; assert r.is_file(), r"
COPY models/ ./models/
RUN mkdir -p /app/data/model-registry
CMD ["python", "scripts/automatic_learner.py"]
