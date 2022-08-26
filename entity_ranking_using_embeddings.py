import json
import requests
import gzip
import sys
from tqdm import tqdm
import os
import argparse
from scipy import spatial
import operator
import utils
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


def get_candidate_entity_set(example: AspectLinkExample) -> Set[str]:
    candidate_aspects: List[Aspect] = example.candidate_aspects
    entities: List[str] = []
    for aspect in candidate_aspects:
        entities.extend(utils.get_entity_ids_only(aspect.aspect_content.entities))
    return set(entities)


def score_entities(
        context_entities: Set[str],
        candidate_entities: Set[str],
        entity_embeddings: Dict[str, List[float]]
) -> Dict[str, float]:
    return {
        entity_id: score_entity(entity_id, context_entities, entity_embeddings)
        for entity_id in candidate_entities
    }


def score_entity(
        target_entity: str,
        context_entities: Set[str],
        entity_embeddings: Dict[str, List[float]]
) -> float:
    if target_entity in entity_embeddings:
        return sum([
            cosine_similarity(entity_embeddings[target_entity], entity_embeddings[context_entity])
            for context_entity in context_entities if context_entity in entity_embeddings
        ])
    return 0.0


def cosine_similarity(emb1, emb2):
    return 1 - spatial.distance.cosine(emb1, emb2)


def make_run_file_strings(query_id: str, scores: Dict[str, float]) -> List[str]:
    return [
        query_id + ' Q0 ' + doc_id + ' ' + str(scores[doc_id]) + ' ' + str(rank + 1) + ' Relatedness'
        for rank, doc_id in enumerate(scores) if scores[doc_id] != 0
    ]


def rank_entities(
        data_file: str,
        context_type: str,
        entity_embeddings: Dict[str, List[float]]
) -> List[str]:
    total = totals[os.path.basename(data_file)]

    run_file_strings: List[str] = []
    for example in tqdm(utils.aspect_link_examples(data_file), total=total):
        query_id: str = example.id
        candidate_entity_set: Set[str] = get_candidate_entity_set(example)
        context_entity_set: Set[str] = utils.get_entity_ids_only(
            example.context.sentence.entities) if context_type == 'sent' else get_entity_ids_only(
            example.context.paragraph.entities)
        entity_scores: Dict[str, float] = score_entities(
            context_entities=context_entity_set,
            candidate_entities=candidate_entity_set,
            entity_embeddings=entity_embeddings
        )
        entity_scores = dict(sorted(entity_scores.items(), key=operator.itemgetter(1), reverse=True))
        run_file_strings.extend(make_run_file_strings(query_id=query_id, scores=entity_scores))

    return run_file_strings


def write_to_file(data: List[str], output_file: str):
    with open(output_file, 'a') as f:
        for d in data:
            f.write("%s\n" % d)


def main():
    parser = argparse.ArgumentParser("Rank entities using semantic relatedness.")
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument("--embeddings", help="Embedding File.", required=True, type=str)
    parser.add_argument("--context", help="Type of context to use (sent|para)", required=True, type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading entity embeddings...')
    with open(args.embeddings, 'r') as f:
        entity_embeddings: Dict[str, List[float]] = json.load(f)
    print('[Done].')

    print('Generating entity ranking...')
    run_file_strings: List[str] = rank_entities(args.data, args.context, entity_embeddings)
    print('[Done].')

    print('Writing to file...')
    write_to_file(run_file_strings, args.save)
    print('[Done].')


if __name__ == '__main__':
    main()
