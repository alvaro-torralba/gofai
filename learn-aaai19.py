#! /usr/bin/env python

import argparse
import os
import os.path
import shutil
import sys
import logging
from lab.environments import  LocalEnvironment
import lab.tools
lab.tools.configure_logging()

import json

sys.path.append(f'{os.path.dirname(__file__)}/training')
import training
from run_experiment import RunExperiment
from aleph_experiment import AlephExperiment
from candidate_models import CandidateModels

from timer import CountdownWCTimer

from partial_grounding_rules import run_step_partial_grounding_rules
from optimize_smac import run_smac_partial_grounding, run_smac_bad_rules, run_smac_search
from instance_set import InstanceSet, select_instances_from_runs,select_instances_from_runs_with_properties
from utils import SaveModel, filter_training_set, combine_training_sets

from downward import suites

# All time limits are in seconds
TIME_LIMITS_SEC = {
    'run_experiment' : 1800, # 30 minutes
    'sklearn-step' : 1800,
}

# Memory limits are in MB
MEMORY_LIMITS_MB = {
    'run_experiment' : 1024*4
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="path to domain file. Alternatively, just provide a path to the directory with a domain.pddl and instance files.")
    parser.add_argument("problem", nargs="*", help="path to problem(s) file. Empty if a directory is provided.")
    parser.add_argument("--domain_knowledge_file", help="path to store knowledge file.")

    parser.add_argument("--path", default='./data', help="path to store results")
    parser.add_argument("--training_data", help="path to an already existing training data folder.")

    parser.add_argument("--cpus", type=int, default=1, help="number of cpus available")
    parser.add_argument("--total_time_limit", default=30, type=int, help="time limit")
    parser.add_argument("--total_memory_limit", default=7*1024, help="memory limit")
    parser.add_argument("--resume", action="store_true", help="if true, do not delete intermediate files (not recommended for final runs)")

    args = parser.parse_args()

    args.domain = os.path.abspath(args.domain)
    args.problem = [os.path.abspath(p) for p in args.problem]
    if args.domain_knowledge_file:
        args.domain_knowledge_file = os.path.abspath(args.domain_knowledge_file)

    return args


def main():
    args = parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))

    TRAINING_DIR=args.path

    REPO_GOOD_OPERATORS = f"{ROOT}/fd-symbolic"
    REPO_LEARNING = f"{ROOT}/learning"
    BENCHMARKS_DIR = f"{TRAINING_DIR}/instances"
    REPO_PARTIAL_GROUNDING = f"{ROOT}/fd-partial-grounding"

    save_model = SaveModel(args.domain_knowledge_file)

    timer = CountdownWCTimer(args.total_time_limit)

    if not args.resume:
        if os.path.exists(TRAINING_DIR):
            shutil.rmtree(TRAINING_DIR)
        os.mkdir(TRAINING_DIR)

    if args.resume and os.path.exists(BENCHMARKS_DIR):
        if os.path.isdir(args.domain): # If the first argument is a folder instead of a domain file
            args.domain += "/domain.pddl"
        pass # TODO: Assert that instances are the same as before
    else:
        # Copy all input benchmarks to the directory
        if os.path.isdir(args.domain): # If the first argument is a folder instead of a domain file
            shutil.copytree(args.domain, BENCHMARKS_DIR)
            args.domain += "/domain.pddl"
        else:
            os.mkdir(BENCHMARKS_DIR)
            shutil.copy(args.domain, BENCHMARKS_DIR)
            for problem in args.problem:
                shutil.copy(problem, BENCHMARKS_DIR)

    ENV = LocalEnvironment(processes=args.cpus)
    SUITE_ALL = suites.build_suite(TRAINING_DIR, ['instances'])

    RUN = RunExperiment (TIME_LIMITS_SEC ['run_experiment'], MEMORY_LIMITS_MB['run_experiment'])

    ###
    # Run lama and symbolic search to gather all training data
    ###
    if not os.path.exists(f'{TRAINING_DIR}/runs-lama'):
        logging.info("Running LAMA on all traning instances (remaining time %s)", timer)
        # Run lama, with empty config and using the alias
        RUN.run_planner(f'{TRAINING_DIR}/runs-lama', REPO_PARTIAL_GROUNDING, [], ENV, SUITE_ALL, driver_options = ["--alias", "lama-first",
                                                                                                                   "--transform-task", f"{REPO_PARTIAL_GROUNDING}/builds/release/bin/preprocess-h2",                                                                                                                   "--transform-task-options", f"h2_time_limit,300"])
    else:
        assert args.resume

    instances_manager = InstanceSet(f'{TRAINING_DIR}/runs-lama')


    if args.training_data:
        instances_manager.add_training_data(args.training_data)
    else:
        # We run the good operators tool only on instances solved by lama in less than 30 seconds
        instances_to_run_good_operators = instances_manager.select_instances([lambda i, p : p['search_time'] < 30])

        SUITE_GOOD_OPERATORS = suites.build_suite(TRAINING_DIR, [f'instances:{name}.pddl' for name in instances_to_run_good_operators])
        if not os.path.exists(f'{TRAINING_DIR}/good-operators-unit'):
            logging.info("Running good operators with unit cost on %d traning instances (remaining time %s)", len(instances_to_run_good_operators), timer)
            RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-unit', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true, cost_type=1)"], ENV, SUITE_GOOD_OPERATORS)
        else:
            assert args.resume
        instances_manager.add_training_data(f'{TRAINING_DIR}/good-operators-unit')


    ####
    # Training of priority partial grounding models
    ####
    ####
    # TRAINING_SET = f'{TRAINING_DIR}/good-operators-unit'

    # aleph_experiment = AlephExperiment(REPO_LEARNING, args.domain, time_limit=TIME_LIMITS_SEC ['train-hard-rules'], memory_limit=MEMORY_LIMITS_MB ['train-hard-rules'])
    # if not os.path.exists(f'{TRAINING_DIR}/partial-grounding-aleph'):
    #     logging.info("Learning Aleph probability model (remaining time %s)", timer)
    #     aleph_experiment.run_aleph_class_probability (f'{TRAINING_DIR}/partial-grounding-aleph', TRAINING_SET, ENV)
    # else:
    #     assert args.resume

    # # TODO: save_model.save()

    if not os.path.exists(f'{TRAINING_DIR}/partial-grounding-sklearn'):
        logging.info("Learning sklearn models (remaining time %s)", timer)
        run_step_partial_grounding_rules(REPO_LEARNING, instances_manager.get_training_datasets(), f'{TRAINING_DIR}/partial-grounding-sklearn', args.domain, time_limit=TIME_LIMITS_SEC['sklearn-step'],
                                         feature_selection_methods=["DT"], good_operator_filenames = ['good_operators'])
    else:
        assert args.resume

    # Save models in individual kw files
    sk_models = [name for name in os.listdir(f'{TRAINING_DIR}/partial-grounding-sklearn') if name.startswith("model")]

    save_model = SaveModel(args.domain_knowledge_file, ext=True)

    for model in sk_models:
        if os.path.exists(os.path.join(TRAINING_DIR, model)):
            shutil.rmtree(os.path.join(TRAINING_DIR, model))
        shutil.copytree(os.path.join(f'{TRAINING_DIR}/partial-grounding-sklearn', model), os.path.join(TRAINING_DIR, model))

        configuration = {'alias' : 'lama-first', 'name' : model}

        for f in os.listdir(f'{TRAINING_DIR}/partial-grounding-sklearn/{model}'):
            if f.endswith('.model'):
                configuration[f"model_{f[:-6]}"] = f

        with open(os.path.join(TRAINING_DIR, model, 'config'), 'w') as config_file:
            json.dump(configuration, config_file)

        save_model.save([ os.path.join(TRAINING_DIR, model)])






if __name__ == "__main__":
    main()
