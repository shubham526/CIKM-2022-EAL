from typing import List, Tuple, Dict, Any
import json
import argparse
import sys


def read_data(data_file):
    with open(data_file, 'r') as f:
        return [json.loads(line) for line in f]


def convert_to_dict(data):
    data_dict = {}
    for d in data:
        context_id = d['context']['id']
        entities = d['entities']
        entity_dict = {entity['entity_id']: entity['entity_desc'].strip() for entity in entities}
        data_dict[context_id] = entity_dict
    return data_dict


def write_to_file(data_dict, save):
    with open(save, 'a') as f:
        for query_id, entities in data_dict.items():
            for entity_id, entity_desc in entities.items():
                f.write("%s\t%s\t%s\n" % (query_id, entity_id, entity_desc))


def main():
    parser = argparse.ArgumentParser("Create entity data file.")
    parser.add_argument("--data", help="Entity data in JSON-L format.", required=True, type=str)
    parser.add_argument("--save", help="Data file to save in TSV format.", required=True, type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading data...')
    data = read_data(args.data)
    print('[Done].')

    print('Concerting to TSV...')
    data_dict = convert_to_dict(data)
    print('[Done].')

    print('Writing to file...')
    write_to_file(data_dict, args.save)
    print('[Done].')


if __name__ == '__main__':
    main()
