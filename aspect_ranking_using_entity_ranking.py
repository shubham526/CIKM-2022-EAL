import json
import gzip
import sys
from utils import tqdm_joblib
from tqdm import tqdm
import os
import argparse
import utils
import operator
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


def rank_aspects(data_file: str, entity_ranking: Dict[str, Dict[str, float]], k: int) -> Dict[str, Dict[str, float]]:
    total = totals[os.path.basename(data_file)]
    ranking: Dict[str, Dict[str, float]] = {}
    for example in tqdm(utils.aspect_link_examples(data_file), total=total):
        query_id = example.id
        if query_id in entity_ranking:
            documents: List[Aspect] = example.candidate_aspects
            entities = entity_ranking[query_id]
            entities = dict(list(entities.items())[:k])
            rank_docs(query_id, documents, entities, ranking)

    return ranking


def write_to_file(ranking: Dict[str, Dict[str, float]], save: str):
    with open(save, 'w') as f:
        for query_id, scores in ranking.items():
            sorted_scores = dict(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))
            rank = 1
            for doc_id, score in sorted_scores.items():
                f.write(query_id + ' Q0 ' + doc_id + ' ' + str(rank) + ' ' + str(score) + ' EntityRanking\n')
                rank += 1


def load_run(entity_run_file: str) -> Dict[str, Dict[str, float]]:
    ranking: Dict[str, Dict[str, float]] = {}
    with open(entity_run_file, 'r') as f:
        for line in f:
            line_parts = line.split()
            query_id = line_parts[0]
            doc_id = line_parts[2]
            doc_score = float(line_parts[4])
            scores: Dict[str, float] = ranking[query_id] if query_id in ranking else {}
            scores[doc_id] = doc_score
            ranking[query_id] = scores

    return ranking


def rank_docs(
        query_id: str,
        candidate_aspects: List[Aspect],
        entity_ranking: Dict[str, float],
        ranking: Dict[str, Dict[str, float]]
) -> None:
    aspect_ranking: Dict[str, float] = {}
    for aspect in candidate_aspects:
        aspect_id: str = aspect.aspect_id
        aspect_entities: List[str] = utils.get_entity_ids_only(aspect.aspect_content.entities)
        aspect_score: float = score_aspect(aspect_entities, entity_ranking)
        aspect_ranking[aspect_id] = aspect_score
    ranking[query_id] = aspect_ranking


def score_aspect(aspect_entities: List[str], entity_scores: Dict[str, float]) -> float:

    c = 0
    for entity_id in entity_scores.keys():
        if entity_id in aspect_entities:
            c += 1
    return float(c)


def main():
    parser = argparse.ArgumentParser("Rank aspects using entity ranking.")
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--entity-run", help="Entity run file.", required=True, type=str)
    parser.add_argument("--save", help="Output directory.", required=True, type=str)
    parser.add_argument("--k", help="Top-K entities to consider.", type=int, default=100)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading entity run file...')
    entity_ranking: Dict[str, Dict[str, float]] = load_run(args.entity_run)
    print('[Done].')

    print('Ranking aspects..')
    ranking: Dict[str, Dict[str, float]] = rank_aspects(args.data, entity_ranking, args.k)
    print('[Done].')

    print('Writing to run file...')
    write_to_file(ranking, args.save)
    print('[Done].')


if __name__ == '__main__':
    main()
