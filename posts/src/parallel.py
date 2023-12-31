from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm


def print_before_sleep(retry_state):
    print(
        f"Retry attempt {retry_state.attempt_number} for operation failed: {retry_state.outcome.exception()}"
    )


@retry(
    wait=wait_exponential(multiplier=5, min=1, max=60),
    stop=stop_after_attempt(5),
    before_sleep=print_before_sleep,
)
def retry_wrapper(func, *args, **kwargs):
    return func(*args, **kwargs)


def parallelize_function(funcs, args_list, kwargs_list=None, max_workers=10):
    if kwargs_list is None:
        kwargs_list = [{}] * len(
            args_list
        )  # Empty dictionaries if no kwargs are provided

    if not isinstance(funcs, list):
        funcs = [funcs]  # Make it a list if a single function is provided

    # Ensure args_list and kwargs_list have the same length
    if len(args_list) != len(kwargs_list):
        raise ValueError("args_list and kwargs_list must have the same length.")

    results = [None] * len(args_list)  # Pre-allocate results list with None values
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_index = {}
        func_iter = cycle(funcs)  # Use itertools.cycle to handle function iteration

        # Submit tasks to the executor
        for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list)):
            func = next(func_iter)
            future = executor.submit(retry_wrapper, func, *args, **kwargs)
            futures_to_index[future] = i  # Map future to its index in args_list

        # Collect results as tasks complete
        for future in tqdm(
            as_completed(futures_to_index),
            total=len(futures_to_index),
            desc="Processing tasks",
        ):
            index = futures_to_index[future]
            try:
                result = future.result()
                results[index] = result  # Place result in the corresponding index
            except Exception as exc:
                print(f"Task {index} generated an exception: {exc}")
                results[index] = exc  # Store the exception in the results list

    return results


# Example usage:
# funcs = [chain1.invoke, chain2.invoke]  # List of functions for load balancing
# args_list = [(arg1,), (arg2,), ...]  # List of argument tuples
# kwargs_list = [{"kwarg1": value1}, {"kwarg2": value2}, ...]  # List of keyword argument dictionaries
# results = parallelize_function(funcs, args_list, kwargs_list)
