FROM python:3.11-slim

ENV CARGO_HOME=/opt/rust/cargo \
    RUSTUP_HOME=/opt/rust/rustup \
    PATH="/opt/rust/cargo/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}" \
    && pip install --upgrade pip \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly \
    && chmod -R 755 /opt/rust

WORKDIR /app

COPY requirements.txt ./backend/
RUN pip install --no-cache-dir -r ./backend/requirements.txt

COPY ./backend ./backend

EXPOSE 8888
ENV PYTHONPATH=/app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "1", "--log-level", "trace"]
