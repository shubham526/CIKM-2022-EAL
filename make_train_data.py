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
    'validation.jsonl.gz': 4313
}
processor = utils.TextProcessor()


def to_pairwise_data(
        example: AspectLinkExample,
        context_type: str,
        desc: Dict[str, Dict[str, str]],
) -> List[str]:
    data: List[str] = []

    query_id: str = example.id
    query: str = processor.preprocess(
        example.context.sentence.content) if context_type == 'sent' else processor.preprocess(
        example.context.paragraph.content)

    if query_id in desc:

        query_desc_dict: Dict[str, str] = desc[query_id]
        pos_entities, neg_entities = utils.get_entities(example.candidate_aspects, example.true_aspect)

        k = min(len(pos_entities), len(neg_entities))
        pos_entities = list(pos_entities)[:k]
        neg_entities = list(neg_entities)[:k]

        entity_pairs: List[List[str]] = [[a, b] for a in pos_entities for b in neg_entities if a != b]

        for pos_entity, neg_entity in entity_pairs:
            if pos_entity in query_desc_dict and neg_entity in query_desc_dict:
                data.append(json.dumps({
                    'query_id': query_id,
                    'query': query,
                    'doc_pos': processor.preprocess(query_desc_dict[pos_entity]),
                    'doc_neg': processor.preprocess(query_desc_dict[neg_entity])
                }))

    return data


def to_pointwise_data(
        example: AspectLinkExample,
        context_type: str,
        desc: Dict[str, Dict[str, str]],
) -> List[str]:
    data: List[str] = []

    query_id: str = example.id
    query: str = processor.preprocess(
        example.context.sentence.content) if context_type == 'sent' else processor.preprocess(
        example.context.paragraph.content)

    if query_id in desc:
        query_desc_dict: Dict[str, str] = desc[query_id]
        pos_entities, neg_entities = utils.get_entities(example.candidate_aspects, example.true_aspect)

        k = min(len(pos_entities), len(neg_entities))
        pos_entities = list(pos_entities)[:k]
        neg_entities = list(neg_entities)[:k]

        for entity_id in pos_entities:
            if entity_id in query_desc_dict:
                data.append(json.dumps({
                    'query_id': query_id,
                    'query': query,
                    'doc': processor.preprocess(query_desc_dict[entity_id]),
                    'label': 1,
                }))

        for entity_id in neg_entities:
            if entity_id in query_desc_dict:
                data.append(json.dumps({
                    'query_id': query_id,
                    'query': query,
                    'doc': processor.preprocess(query_desc_dict[entity_id]),
                    'label': 0,
                }))

    return data


def create_data(
        data_type: str,
        data_file: str,
        save: str,
        context_type: str,
        desc_dict: Dict[str, Dict[str, str]],
        num_workers: int
) -> None:
    print('Data type: {}'.format(data_type))
    print('Context type: {}'.format(context_type))
    print('Number of processes = {}'.format(num_workers))
    total = totals[os.path.basename(data_file)]

    if data_type == 'pairwise':
        with tqdm_joblib(tqdm(desc="Progress", total=total)) as progress_bar:
            data = Parallel(n_jobs=num_workers, backend='multiprocessing')(
                delayed(to_pairwise_data)(example, context_type, desc_dict)
                for example in utils.aspect_link_examples(data_file))
    elif data_type == 'pointwise':
        with tqdm_joblib(tqdm(desc="Progress", total=total)) as progress_bar:
            data = Parallel(n_jobs=num_workers, backend='multiprocessing')(
                delayed(to_pointwise_data)(example, context_type, desc_dict)
                for example in utils.aspect_link_examples(data_file))
    else:
        raise ValueError('Mode must be `pairwise` or `pointwise`.')

    print('Writing to file...')
    for d in data:
        utils.write_to_file(d, save)
    print('[Done].')
    print('File written to ==> {}'.format(save))


def main():
    parser = argparse.ArgumentParser("Create a training file.")
    parser.add_argument("--mode", help="Type of data (pairwise|pointwise).", required=True, type=str)
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--save", help="Output directory.", required=True, type=str)
    parser.add_argument("--context", help="Type of context to use (sent|para). Default: paragraph context.",
                        required=True, type=str)
    parser.add_argument("--desc", help='File containing entity description.', required=True, type=str)
    parser.add_argument("--num-workers", help="Number of processes to use. Default: 4.",
                        default=4, type=int)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    if args.mode == 'pairwise':
        print('Creating pairwise data.')
    elif args.mode == 'pointwise':
        print('Creating pointwise data.')
    else:
        raise ValueError('Task must be `pairwise` or `pointwise`.')

    print('Reading description file....')
    desc_dict: Dict[str, Dict[str, str]] = utils.read_entity_data_file(args.desc)
    print('[Done].')

    save: str = args.save + '/' + 'train.' + args.mode + '.jsonl'

    create_data(
        data_type=args.mode,
        data_file=args.data,
        save=save,
        context_type=args.context,
        desc_dict=desc_dict,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
