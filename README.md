# Dynamic Batching Inference System

A high-performance inference server implementation using **FastAPI** and **PyTorch**, featuring a dynamic batching mechanism to optimize throughput and latency.

## Architecture

- **Server**: FastAPI application serving HTTP endpoints.
- **Dynamic Batcher**: Collects incoming requests into batches based on `max_batch_size` or `max_latency`.
- **Worker**: Executes the PyTorch model on the collected batches.
- **Model**: Plug-and-play architecture (currently using `prajjwal1/bert-tiny` for demonstration, easily switchable to `distilbert` or others).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Server**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:.
    python src/api/server.py
    ```

    Configuration via environment variables:
    - `MAX_BATCH_SIZE`: Default 32
    - `MAX_LATENCY_MS`: Default 10.0 (ms)
    - `MODEL_NAME`: Default `prajjwal1/bert-tiny`

3.  **Run Benchmarks**:
    ```bash
    python benchmarks/benchmark.py -n 100 -c 20
    ```

## Project Structure

```
src/
  api/          # FastAPI server and endpoints
  core/         # Core logic (Batcher, Request Queue)
  models/       # Model wrappers (NLP, Base)
benchmarks/     # Load testing scripts
tests/          # Unit tests
```

## Key Concepts Demonstrated

- **Request Queuing**: `asyncio.Queue` handles incoming traffic bursts.
- **Latency vs Throughput**: `max_latency` creates a time window to fill batches, improving hardware utilization at the cost of slight latency increase.
- **Concurrency**: `asyncio` handles I/O bound web requests, while `run_in_executor` offloads compute-bound inference.
