import json
import gzip
import sys
from utils import tqdm_joblib
from tqdm import tqdm
import os
import argparse
import utils
from joblib import Parallel, delayed
from typing import List, Dict, Set, Tuple, Any
from object_models import Location, Entity, AnnotatedText, AspectLinkExample, Aspect, Context

totals = {
    'nanni-test.jsonl.gz': 18289,
    'overly-frequent.jsonl.gz': 429160,
    'test.jsonl.gz': 4967,
    'train-remaining.jsonl.gz': 544892,
    'train-small.jsonl.gz': 5498,
    'validation.jsonl.gz': 4313,
    'nanni-201.modified.jsonl.gz': 143
}

def create_queries(data, context_type, save):
    d = {}
    total = totals[os.path.basename(data)]
    for example in tqdm(utils.aspect_link_examples(data),total=total):
        query_id = example.id
        query = example.context.sentence.content if context_type == 'sent' else example.context.paragraph.content
        d[query_id] = query

    print('Writing to file...')
    write_to_file(d, save)
    print('[Done].')


def write_to_file(data, save):
    with open(save, 'a') as f:
        for query_id, query in data.items():
            f.write("%s\t%s\n" % (query_id, query))


def main():
    parser = argparse.ArgumentParser("Create a queries file.")
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--context", help="Context type (sent|para).", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    create_queries(data=args.data, context_type=args.context, save=args.save)

if __name__ == '__main__':
    main()
