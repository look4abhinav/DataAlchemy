# Start with Python 3.13 slim base image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy project files
COPY pyproject.toml README.md ./
COPY main.py ./
COPY prompts ./prompts

# Create and activate virtual environment using UV
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Install project dependencies using UV
RUN uv sync --upgrade


# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set up Streamlit configuration
RUN mkdir -p /home/appuser/.streamlit
COPY <<EOF /home/appuser/.streamlit/config.toml
[server]
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#2E5AAC"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFF"
textColor = "#1E3A8A"
EOF

# Expose Streamlit port
EXPOSE 8501

# Set entrypoint
ENTRYPOINT ["streamlit", "run"]

# Set default command
CMD ["main.py"]