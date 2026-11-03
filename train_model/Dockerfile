FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JUPYTER_ENABLE_LAB=yes

WORKDIR /workspace

# Base dependencies for notebook execution and model training.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc pandoc texlive-fonts-recommended texlive-latex-extra texlive-plain-generic texlive-xetex \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies used by the notebook and notebook server.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir jupyterlab ipykernel

# Copy train_model under its own directory so notebook-relative paths keep working.
COPY ./train_model /workspace/train_model
RUN ln -sfn /workspace/train_model/datasets /workspace/datasets

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.root_dir=/workspace"]
