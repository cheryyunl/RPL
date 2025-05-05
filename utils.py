# utils.py
import math
import pandas as pd
import json
import os

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    """Get the k-th chunk of a list divided into n chunks"""
    chunks = split_list(lst, n)
    return chunks[k]

def dump_to_jsonl(obj: list[dict], path: str):
    """Write a list of dictionaries to a jsonl file"""
    with open(path, 'w') as file:
        file