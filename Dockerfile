###############
# BUILD IMAGE #
###############

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Copy the required uv files to create the virtual environment
ADD pyproject.toml .python-version uv.lock /app/
WORKDIR /app

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

FROM builder AS runtime

# activate the project virtual environment by placing its binary directory at the front of the path 
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="$PYTHONPATH:/app/src"

# add the source code last (avoid rebuilding venv on every change)
ADD . /app

# Expose the port that the application will run on
EXPOSE 8501

ENTRYPOINT exec streamlit run ./src/streamlit_apps/Welcome.py
