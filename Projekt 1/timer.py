import time

times = {}

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        increment_times(func.__name__, end - start)
        return result
    return wrapper

def increment_times(func_name: str, exec_time: float) -> None:
    if times.get(func_name, None) is None:
        times[func_name] = {
            "executed" : 0,
            "total_time": 0
        }
        
    times[func_name]["executed"] += 1
    times[func_name]["total_time"] += exec_time

def print_times():
    total_time = 0
    for func_name, data in times.items():
        avg_time = 0.0 if data["executed"] == 0 else data["total_time"] / data["executed"]
        total_time += data['total_time']
        print(f"{func_name} executed {data['executed']} times. Total time: {data['total_time']}. Avg Time: {avg_time}")
    
    avg_per_func = total_time / len(times) if len(times) != 0 else 0.0
    print(f"Total Time: {total_time}. Avg per Func: {avg_per_func}")
    