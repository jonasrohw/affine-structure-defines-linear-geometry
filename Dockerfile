# Immutable multi-platform manifests, resolved from the registries on 2026-10-06.
FROM ghcr.io/astral-sh/uv:0.12.0@sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105 AS uv
FROM python:3.13.11-slim-bookworm@sha256:20080e807bfc404f8450b185cf0fc95d553462673598549613735f70a5b4d5d0

# Freeze the OS package universe as well as Python wheels. Snapshot metadata has
# intentionally expired; its content and package hashes remain immutable.
RUN rm -f /etc/apt/sources.list.d/debian.sources \
    && printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20261001T000000Z bookworm main\n' > /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libcairo2 fonts-dejavu-core latexmk texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra \
    && dpkg-query -W > /opt/os-packages.txt \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    SOURCE_DATE_EPOCH=1791244800 \
    TZ=UTC
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-install-project --no-dev
COPY experiments/factual_recall experiments/factual_recall
COPY paper paper
RUN uv sync --locked --no-dev
ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "reproduce"]
CMD []
