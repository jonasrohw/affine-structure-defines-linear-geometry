# The root Dockerfile pins this base's Python/uv images and Debian snapshot.
FROM factual-recall:reproducible
# CPython standalone BUILD=20260114, the recovered beta interpreter build.
ADD https://releases.astral.sh/github/python-build-standalone/releases/download/20260114/cpython-3.14.2%2B20260114-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz /tmp/python.tar.gz
RUN echo 'b54c8ad23ded6b9a2e9da754bf3a5e622763e3ac224db555a6a8888f40897036  /tmp/python.tar.gz' | sha256sum -c - \
    && mkdir /opt/beta-python \
    && tar -xzf /tmp/python.tar.gz -C /opt/beta-python --strip-components=1 \
    && rm /tmp/python.tar.gz
COPY pyproject.toml uv.lock report-requirements.txt /opt/beta-environment/
ENV UV_PROJECT_ENVIRONMENT=/opt/beta-venv
RUN cd /opt/beta-environment \
    && uv sync --frozen --no-install-project --extra plots --extra dev --python /opt/beta-python/bin/python3.14 \
    && uv pip install --python /opt/beta-venv/bin/python --require-hashes --no-deps -r /opt/beta-environment/report-requirements.txt \
    && uv cache clean
# Required also when reusing the earlier CPU base from the investigation.
RUN apt-get update && apt-get install -y --no-install-recommends cm-super dvipng \
    && dpkg-query -W > /opt/os-packages.txt \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/app/experiments
