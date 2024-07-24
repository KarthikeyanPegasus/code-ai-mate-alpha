import time
from functools import wraps

def rate_limiter(max_per_minute):
    def decorator(func):
        timestamps = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal timestamps
            now = time.time()
            # Filter out timestamps outside of the last minute
            timestamps = [t for t in timestamps if now - t < 60]
            if len(timestamps) < max_per_minute:
                timestamps.append(now)
                return func(*args, **kwargs)
            else:
                # Calculate sleep time to maintain the rate limit
                sleep_time = 60 - (now - timestamps[0])
                print(f"Rate limit exceeded. Waiting for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                return wrapper(*args, **kwargs)
        return wrapper
    return decorator
