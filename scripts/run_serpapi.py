import argparse
import json
import os
from serpapi import GoogleSearch
from tqdm import tqdm

base_params = {
  "api_key": os.environ['SERP_API_KEY'],
  "engine": "google",
  "location": "United States",
  "google_domain": "google.com",
  "gl": "us",
  "hl": "en",
  "safe": "active"
}


def main(args):
    with open(args.queries_file) as fin:
        queries = [line.strip() for line in fin]
    print(f"Loaded {len(queries)} queries from {args.queries_file}.")
    assert len(queries) == len(set(queries))  # ensure each query is unique

    search_outputs = {}
    for query in tqdm(queries, desc="Running Search"):
        params = base_params.copy()
        params["q"] = f"write a bio for {query}"
        search = GoogleSearch(params)
        one_result = search.get_dict()
        search_outputs[query] = one_result
    
    with open(args.output_file, 'w') as fout:
        json.dump(search_outputs, fout, indent=2)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    args_ = parser.parse_args()
    main(args_)