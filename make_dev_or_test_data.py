import json
import gzip
import sys
from tqdm import tqdm
import os
import argparse
import utils
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
processor = utils.TextProcessor()


def to_data(
        example: AspectLinkExample,
        context_type: str,
        desc: Dict[str, Dict[str, str]],
) -> List[str]:
    data: List[str] = []

    query_id: str = example.id
    query: str = example.context.sentence.content if context_type == 'sent' else example.context.paragraph.content

    if query_id in desc:
        query_desc_dict: Dict[str, str] = desc[query_id]
        pos_entities, neg_entities = utils.get_entities(example.candidate_aspects, example.true_aspect)

        pos_entities = list(pos_entities)
        neg_entities = list(neg_entities)

        for entity_id in pos_entities:
            if entity_id in query_desc_dict:
                data.append(json.dumps({
                    'query_id': query_id,
                    'query': processor.preprocess(query),
                    'doc_id': entity_id,
                    'doc': processor.preprocess(query_desc_dict[entity_id]),
                    'label': 1,
                }))

        for entity_id in neg_entities:
            if entity_id in query_desc_dict:
                data.append(json.dumps({
                    'query_id': query_id,
                    'query': processor.preprocess(query),
                    'doc_id': entity_id,
                    'doc': processor.preprocess(query_desc_dict[entity_id]),
                    'label': 0,
                }))

    return data


def create_data(
        data_file: str,
        save: str,
        context_type: str,
        desc_dict: Dict[str, Dict[str, str]],
        num_workers: int
) -> None:
    print('Context type: {}'.format(context_type))
    print('Number of processes = {}'.format(num_workers))
    total = totals[os.path.basename(data_file)]

    data = []

    for example in tqdm(utils.aspect_link_examples(data_file), total=total):
        data.append(to_data(example, context_type, desc_dict))

    print('Writing to file...')
    for d in data:
        utils.write_to_file(d, save)
    print('[Done].')
    print('File written to ==> {}'.format(save))


def main():
    parser = argparse.ArgumentParser("Create a dev/test file.")
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument("--context", help="Type of context to use (sent|para)", required=True, type=str)
    parser.add_argument("--desc", help='File containing entity description.', required=True, type=str)
    parser.add_argument("--num-workers", help="Number of processes to use. Default: 4.",
                        default=4, type=int)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Reading description file....')
    desc_dict: Dict[str, Dict[str, str]] = utils.read_entity_data_file(args.desc)
    print('[Done].')

    create_data(
        data_file=args.data,
        save=args.save,
        context_type=args.context,
        desc_dict=desc_dict,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
