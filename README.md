# HF Downloader

A Streamlit application for downloading models from HuggingFace.

## Requirements

- Python 3.11+
- HuggingFace account and API token

## Setup

1. Clone the repository

2. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

3. Add your HuggingFace token to `.env`:
   ```
   HF_TOKEN=your_token_here
   ```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run with Docker

Build the image:
```bash
docker build -t hf-downloader .
```

Run the container:
```bash
docker run --env-file .env -p 8501:8501 hf-downloader
```

Access the app at http://localhost:8501
