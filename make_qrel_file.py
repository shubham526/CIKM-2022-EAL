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

def get_entities(candidate_aspects: List[Aspect], true_aspect: str) -> Tuple[Set[str], Set[str]]:
    pos_entities: Set[str] = set()
    neg_entities: List[str] = []
    for aspect in candidate_aspects:
        if aspect.aspect_id == true_aspect:
            pos_entities = set(utils.get_entity_ids_only(aspect.aspect_content.entities))
        else:
            neg_entities.extend(utils.get_entity_ids_only(aspect.aspect_content.entities))
    return pos_entities, set(neg_entities)

def create_qrels(data, save):
    run_strings = []
    total = totals[os.path.basename(data)]
    for example in tqdm(utils.aspect_link_examples(data), total=total):
        query_id: str = example.id
        pos_entities, neg_entities = get_entities(example.candidate_aspects, example.true_aspect)
        pos_entities = set(pos_entities)
        for entity_id in pos_entities:
            if entity_id:
                run_string = query_id + ' Q0 ' + entity_id + ' 1'
                run_strings.append(run_string)

    print('Writing to file...')
    write_to_file(run_strings, save)
    print('[Done].')


def write_to_file(data: List[str], output_file: str):
    with open(output_file, 'a') as f:
        for line in data:
            f.write("%s\n" % line)


def main():
    parser = argparse.ArgumentParser("Create a qrels file.")
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    create_qrels(data=args.data, save=args.save)

if __name__ == '__main__':
    main()