FROM python:3.12-slim

ENV PYTHONHASHSEED=0 \
    MPLBACKEND=Agg

WORKDIR /artifact
COPY requirements-lock.txt pyproject.toml README.md LLM_STUDY.md ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY scripts ./scripts
COPY specs ./specs
RUN python -m pip install --no-cache-dir -r requirements-lock.txt && \
    python -m pip install --no-cache-dir --no-deps -e .

CMD ["bash", "scripts/reproduce_quick.sh"]
