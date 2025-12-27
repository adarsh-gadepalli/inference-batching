import asyncio
import aiohttp
import time
import statistics
import argparse
import random

async def send_request(session, url, text):
    start = time.time()
    async with session.post(url, json={"text": text}) as response:
        await response.json()
        return time.time() - start

async def benchmark(url, num_requests, concurrency):
    print(f"starting benchmark: {num_requests} requests, concurrency {concurrency}")
    
    texts = [
        "this movie is great!",
        "i didn't like this product.",
        "the weather is amazing today.",
        "customer service was terrible.",
        "i am neutral about this."
    ]
    
    tasks = []
    # limit the number of open connections at once
    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        start_total = time.time()
        
        for _ in range(num_requests):
            text = random.choice(texts)
            tasks.append(send_request(session, url, text))
            
        # fire all requests!
        latencies = await asyncio.gather(*tasks)
        total_time = time.time() - start_total
        
    # calculate stats
    avg_latency = statistics.mean(latencies) * 1000
    p50 = statistics.median(latencies) * 1000
    p95 = statistics.quantiles(latencies, n=20)[18] * 1000
    throughput = num_requests / total_time
    
    print(f"\nresults:")
    print(f"total time: {total_time:.2f}s")
    print(f"throughput: {throughput:.2f} req/s")
    print(f"latency (avg): {avg_latency:.2f}ms")
    print(f"latency (p50): {p50:.2f}ms")
    print(f"latency (p95): {p95:.2f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("-n", "--num", type=int, default=100)
    parser.add_argument("-c", "--concurrency", type=int, default=10)
    args = parser.parse_args()
    
    asyncio.run(benchmark(args.url, args.num, args.concurrency))

