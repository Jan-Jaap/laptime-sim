###############
# BUILD IMAGE #
###############

# FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
# Copy the entire project
ADD . /app
WORKDIR /app

ENV UV_LINK_MODE=copy

# Install dependencies (runtime only)
RUN uv sync --frozen --no-install-project --no-dev
# RUN uv sync --frozen --no-install-project

# activate the project virtual environment by placing its binary directory at the front of the path 
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port that the application will run on
EXPOSE 3801

CMD ["streamlit", "run", "./src/streamlit_apps/Welcome.py"]

