## ------------------------------- Builder Stage ------------------------------ ## 
FROM python:3.12-bookworm AS builder

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download the latest installer, install it and then remove it
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh

# Set up the UV environment path correctly
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY ./pyproject.toml .

RUN uv sync --no-dev

## ------------------------------- Production Stage ------------------------------ ##
FROM python:3.12-slim-bookworm AS production

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY /resources resources
COPY /src src

# Set up environment variables for production
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="$PYTHONPATH:/app/src"

# Expose the specified port for Streamlit
EXPOSE $PORT

# ENTRYPOINT exec streamlit run ./src/streamlit_apps/Welcome.py
CMD ["streamlit", "run", "./src/streamlit_apps/Welcome.py"]
