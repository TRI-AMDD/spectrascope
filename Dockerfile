# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Build stage: compilers and git stay here so their libs (e.g. libbrotli1) are not in the final image.
FROM base as build

# Install build-essential which includes gcc
RUN apt-get update && apt-get install -y build-essential gfortran git && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN python -m pip install -U pip
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt
RUN git clone --branch main https://github.com/AmalyaJohnson-TRI/SmartProcessAnalytics.git
RUN python -m pip install -e SmartProcessAnalytics

# Copy the source code into the container.
COPY app app
COPY spectrascope spectrascope
COPY tests tests
COPY setup.py pyproject.toml requirements.txt README.md ./
RUN python -m pip install -e .

# Final stage: copy the installed Python env and source from the build stage.
FROM base
COPY --from=build /usr/local /usr/local
COPY --from=build --chown=appuser:appuser /app /app

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application.
WORKDIR /app/app
ENTRYPOINT ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
