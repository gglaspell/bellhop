FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-tk \
    liboctomap-dev \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python venv (avoids PEP 668 externally-managed-environment error) ──────
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ── Application source ─────────────────────────────────────────────────────
WORKDIR /app
COPY pipelines/ ./pipelines/
COPY cli.py .
COPY gui.py .

# Default: CLI mode. Override with:
#   docker run ... bellhop --gui          (launch GUI)
#   docker run ... bellhop mesh bag ./out (run a pipeline directly)
ENTRYPOINT ["/opt/venv/bin/python", "/app/cli.py"]
