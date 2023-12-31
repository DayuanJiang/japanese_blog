from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

import ipadic
import MeCab
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

# download japanese stopwords
url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ja/master/stopwords-ja.txt"
stopwords = requests.get(url).text.split("\n")


# parser = MeCab.Tagger("-Owakati")
def extract_nouns_verbs(text):
    parser = MeCab.Tagger(ipadic.MECAB_ARGS)
    parsed_text = parser.parse(text)

    lines = parsed_text.split("\n")
    nouns_verbs = []

    for line in lines:
        if "名詞" in line or "動詞" in line or "形状詞" in line:
            parts = line.split("\t")
            word = parts[0]
            if not word.isascii():
                nouns_verbs.append(word)
    return nouns_verbs


def preprocess(text):
    token_list = [
        token for token in extract_nouns_verbs(text) if token not in stopwords
    ]
    return " ".join(token_list)


# Define a retry decorator with exponential backoff
@retry(wait=wait_exponential(multiplier=5, min=1, max=60), stop=stop_after_attempt(5))
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
