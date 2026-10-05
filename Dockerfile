FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /windpfn
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["windpfn-score"]
