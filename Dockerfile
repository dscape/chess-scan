FROM node:22-alpine AS web-build
WORKDIR /app/web
COPY web/package.json web/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHESS_SCAN_DATA_DIR=/app/data \
    CHESS_SCAN_MODEL_DIR=/app/models \
    CHESS_SCAN_WEB_DIST=/app/web-dist
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY server/ ./server/
RUN pip install --no-cache-dir .
COPY models/ ./models/
COPY --from=web-build /app/web/dist ./web-dist/
RUN mkdir -p /app/data
EXPOSE 8000
CMD ["uvicorn", "chess_scan.main:app", "--app-dir", "server", "--host", "0.0.0.0", "--port", "8000"]
