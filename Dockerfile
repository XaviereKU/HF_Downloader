FROM python:3.11-slim

RUN apt-get update && apt-get install -y vim curl wget net-tools iputils-ping

RUN curl -LsSf https://astral.sh/uv/install.sh | bash
RUN curl -LsSf https://hf.co/cli/install.sh | bash

WORKDIR /app

COPY requirements.txt .

RUN /root/.local/bin/uv pip install --no-cache-dir --system -r requirements.txt

COPY . .

# HF_TOKEN should be passed at runtime via --env-file or -e flag

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]