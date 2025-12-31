import asyncio
import time
from typing import List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class InferenceRequest:
    # simple container for a request, holds the input and the future result
    request_id: str
    input_data: Any # Can be text or Tensor now
    # future is a placeholder for a result that hasn't arrived yet
    future: asyncio.Future = field(default_factory=asyncio.Future)
    arrival_time: float = field(default_factory=time.time)

class DynamicBatcher:

    # dynamically collects requests and runs them in batches
    def __init__(self, model, max_batch_size: int = 32, max_latency_ms: float = 10.0, max_queue_size: int = 10000):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_latency = max_latency_ms / 1000.0  
        # asyncio.queue is thread-safe and async-friendly, perfect for buffering requests
        # bounded queue prevents infinite memory growth under heavy load
        self.queue = asyncio.Queue(maxsize=max_queue_size)  
        self.running = False
        self._loop_task = None

    async def start(self):
        # loads model and starts the background worker task
        self.model.load()
        self.running = True
        # asyncio.create_task fires off a function to run in the background immediately
        self._loop_task = asyncio.create_task(self._process_batches())
        print(f"dynamic batcher started")

    async def stop(self):
        # stops the background worker 
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                # await the cancelled task to let it clean up properly
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def predict(self, input_data: Any) -> Any:
        # puts data in queue and awaits answer
        if self.queue.full():
            raise Exception("server overloaded: queue full")
            
        request = InferenceRequest(
            request_id=str(time.time()),
            input_data=input_data
        )
        await self.queue.put(request)
        # await request.future 'pauses' this function until the batch processor sets the result
        return await request.future

    async def _process_batches(self):
        # infinite loop running in background to form batches
        batch = []
        
        while self.running:
            try:
                # if we have nothing, wait until a request enters the queue
                if not batch:
                    req = await self.queue.get()
                    batch.append(req)
                    
                # once we have a request, start a deadline for the batch
                deadline = batch[0].arrival_time + self.max_latency
                
                
                time_remaining = deadline - time.time()
                
                # keep checking for more items until batch is full or time is up
                while len(batch) < self.max_batch_size and time_remaining > 0:
                    try:
                        # wait_for throws an error if item is not found in time
                        req = await asyncio.wait_for(self.queue.get(), timeout=time_remaining)
                        batch.append(req)
                        time_remaining = deadline - time.time()
                    except asyncio.TimeoutError:
                        # if timeout, break the loop and run batch
                        break
                
                # run the batch if it's not empty
                if batch:
                    await self._run_inference(batch)
                    # clear the batch for the next iteration
                    batch = []
                    
            except Exception as e:
                print(f"error in batch loop: {e}")
                await asyncio.sleep(0.1)

    async def _run_inference(self, batch: List[InferenceRequest]):
        # actually runs inference in the model
        inputs = [req.input_data for req in batch]
        
        # calculate waste metric:
        # heuristic: use tensor shape or string len
        
        if hasattr(inputs[0], 'size'):
             # it's a tensor (pre-tokenized)
             lengths = [t.size(0) for t in inputs]
        else:
             # it's a string
             lengths = [len(s) for s in inputs] 
             
        max_len = max(lengths) if lengths else 0
        total_capacity = len(batch) * max_len
        actual_content = sum(lengths)
        
        waste_ratio = 0.0
        if total_capacity > 0:
            waste_ratio = 1.0 - (actual_content / total_capacity)
            
        try:
            # get the current event loop 
            loop = asyncio.get_running_loop()
            
            # run_in_executor moves model.predict call to separate thread
            results = await loop.run_in_executor(None, self.model.predict, inputs)
            
            # return answers back to the original requests
            for req, result in zip(batch, results):
                if not req.future.done():
                    # return tuple (result, waste_ratio)
                    req.future.set_result((result, waste_ratio))
                    
        except Exception as e:
            # if the batch fails, propagate the error to the requests
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
