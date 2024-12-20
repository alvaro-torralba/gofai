#! /usr/bin/env python

import argparse
import os
import os.path
import shutil
import sys
import logging
from pathlib import Path

import lab.tools

lab.tools.configure_logging()

import json

sys.path.append(f'{os.path.dirname(__file__)}/training')
import training
from aleph_experiment import AlephExperiment
from candidate_models import CandidateModels

from partial_grounding_rules import run_step_partial_grounding_rules
from instance_set import InstanceSet, select_instances_from_runs
from utils import SaveModel, filter_training_set, combine_training_sets

from downward import suites


# from evaluate_model import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("training_data", help="path to an already existing training data folder.")
    parser.add_argument("--domain_knowledge_file", help="path to store knowledge file.")
    parser.add_argument("--path", default='./data', help="path to store results")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ROOT = os.path.dirname(os.path.abspath(__file__))
    REPO_LEARNING = f"{ROOT}/learning"

    save_model = SaveModel(args.domain_knowledge_file)

    TRAINING_DATA = Path(args.training_data)
    DOMAIN_FILE = TRAINING_DATA / "tasks/domain.pddl"
    # args.domain += "/domain.pddl"

    instances_manager = InstanceSet(TRAINING_DATA / "good-operators-unit")
    instances_manager.add_training_data(TRAINING_DATA / "good-operators-unit")

    # aleph_experiment = AlephExperiment(REPO_LEARNING, args.domain, time_limit=TIME_LIMITS_SEC ['train-hard-rules'], memory_limit=MEMORY_LIMITS_MB ['train-hard-rules'])
    # if not os.path.exists(f'{TRAINING_DIR}/partial-grounding-aleph'):
    #     logging.info("Learning Aleph probability model (remaining time %s)", timer)
    #     aleph_experiment.run_aleph_class_probability (f'{TRAINING_DIR}/partial-grounding-aleph', TRAINING_SET, ENV)
    # else:
    #     assert args.resume

    # # TODO: save_model.save()
    TRAINING_DIR = Path(args.path)
    if not os.path.exists(TRAINING_DIR):
        os.makedirs(TRAINING_DIR)

    if os.path.exists(f'{TRAINING_DIR}/partial-grounding-sklearn'):
        shutil.rmtree(f'{TRAINING_DIR}/partial-grounding-sklearn')
    logging.info("Learning sklearn models")
    run_step_partial_grounding_rules(REPO_LEARNING, instances_manager.get_training_datasets(),
                                     f'{TRAINING_DIR}/partial-grounding-sklearn', DOMAIN_FILE, time_limit=60000,
                                     feature_selection_methods=["DT"], good_operator_filenames=['good_operators'])

    # Save models in individual kw files
    sk_models = [name for name in os.listdir(f'{TRAINING_DIR}/partial-grounding-sklearn') if name.startswith("model")]

    save_model = SaveModel(args.domain_knowledge_file, ext=True)

    for model in sk_models:
        if os.path.exists(os.path.join(TRAINING_DIR, model)):
            shutil.rmtree(os.path.join(TRAINING_DIR, model))
        shutil.copytree(os.path.join(f'{TRAINING_DIR}/partial-grounding-sklearn', model),
                        os.path.join(TRAINING_DIR, model))

        configuration = {'alias': 'lama-first', 'name': model}

        for f in os.listdir(f'{TRAINING_DIR}/partial-grounding-sklearn/{model}'):
            if f.endswith('.model'):
                configuration[f"model_{f[:-6]}"] = f

        with open(os.path.join(TRAINING_DIR, model, 'config'), 'w') as config_file:
            json.dump(configuration, config_file)

        save_model.save([os.path.join(TRAINING_DIR, model)])

    # for model in sk_models:
    # evaluate_model(model, args.training_data)


if __name__ == "__main__":
    main()
