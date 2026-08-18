FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN useradd --uid 65532 --create-home --shell /usr/sbin/nologin sandbox
COPY container_worker.py /opt/retool/container_worker.py
RUN chmod 0555 /opt/retool/container_worker.py

USER 65532:65532
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-B", "/opt/retool/container_worker.py"]
