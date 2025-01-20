###############
# BUILD IMAGE #
###############

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Copy the required uv files to create the virtual environment
ADD pyproject.toml .python-version uv.lock /app/
WORKDIR /app

ENV UV_LINK_MODE=copy

# Install dependencies (runtime only)
RUN uv sync --frozen --no-install-project --no-dev
# RUN uv sync --frozen --no-install-project

# activate the project virtual environment by placing its binary directory at the front of the path 
ENV PATH="/app/.venv/bin:$PATH"

# add the source code last (avoid rebuilding venv on every change)
ADD . /app

# Expose the port that the application will run on
EXPOSE 8501

CMD ["streamlit", "run", "./src/streamlit_apps/Welcome.py"]

