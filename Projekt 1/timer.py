import time
import numpy as np
from collections import defaultdict

_timer_enable = False
_timer_metrics = {
    "func_times": defaultdict(lambda: {"executed": 0, "total_time": 0.0}),
    "overhead_time": 0.0
}

def timer(func):
    global _timer_enable
    if not _timer_enable:
        return func

    def wrapper(*args, **kwargs):        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        duration = end - start

        start = time.perf_counter()
        _increment_function_time(func.__qualname__, duration)
        end = time.perf_counter()

        _add_timer_overhead(end - start)
        return result
    return wrapper

def _increment_function_time(func_name: str, duration: float) -> None:
    global _timer_metrics
    _timer_metrics["func_times"][func_name]["executed"] += 1
    _timer_metrics["func_times"][func_name]["total_time"] += duration

def _add_timer_overhead(duration: float) -> None:
    global _timer_metrics
    _timer_metrics["overhead_time"] += duration

def enable_timers() -> None:
    global _timer_enable
    _timer_enable = True

def diable_timers() -> None:
    global _timer_enable
    _timer_enable = False

def print_times() -> None:
    global _timer_enable, _timer_metrics

    if not _timer_enable:
        print("Timer is disabled.")
        return

    funcs_times_metrics = _timer_metrics["func_times"]
    overhead_time = _timer_metrics["overhead_time"]

    if not funcs_times_metrics:
        print("No data to show.")
        print(f"Timer overhead (s): {overhead_time:.6f}")
        return

    func_names = list(funcs_times_metrics.keys())
    func_data = list(funcs_times_metrics.values())

    func_times = np.array([data['total_time'] for data in func_data])
    executions = np.array([data['executed'] for data in func_data])
    avg_times = np.divide(func_times, executions, out=np.zeros_like(func_times), where=executions != 0)

    total_time = np.sum(func_times) + overhead_time

    if total_time == 0:
        percentages = np.zeros_like(func_times)
    else:
        percentages = (func_times * 100.0) / total_time

    func_idxs = np.argsort(percentages)[::-1]

    headers = ["No.", "Func Name", "Execution", "Avg Time (s)", "Total Time (s)", "Percent (%)"]

    display_data = []
    max_lens = [len(header) for header in headers]

    for i, func_idx in enumerate(func_idxs):
        num_text = f"{i + 1}."
        func_name_text = func_names[func_idx]
        execution_text = f"{executions[func_idx]}"
        avg_time_text = f"{avg_times[func_idx]:.6f}"
        func_time_text = f"{func_times[func_idx]:.6f}"
        percentage_text = f"{percentages[func_idx]:.2f}"

        row_data = [num_text, func_name_text, execution_text, avg_time_text, func_time_text, percentage_text]
        display_data.append(row_data)

        for j, text in enumerate(row_data):
            max_lens[j] = max(max_lens[j], len(text))

    max_lens = [length + 4 for length in max_lens]

    header_line = "".join(f"{header:<{max_lens[i]}}" for i, header in enumerate(headers))
    print(header_line)

    for row in display_data:
        data_line = "".join(f"{text:<{max_lens[i]}}" for i, text in enumerate(row))
        print(data_line)

    overhead_percentage = (overhead_time * 100.0) / total_time if total_time > 0 else 0.0
    
    print(f"Total Time (s): {total_time:.6f}")
    print(f"Timer overhead (s): {overhead_time:.6f}")
    print(f"Timer overhead percentage (%): {overhead_percentage:.2f}")