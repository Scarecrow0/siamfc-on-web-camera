import json
import os
from collections import namedtuple


def parse_arguments(in_hp=None, in_evaluation=None, in_run=None):
    if in_hp is None:
        in_hp = {}
    if in_evaluation is None:
        in_evaluation = {}
    if in_run is None:
        in_run = {}
    dir = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(dir, 'parameters', 'hyperparams.json')) as json_file:
        hp = json.load(json_file)
    with open(os.path.join(dir, 'parameters', 'evaluation.json')) as json_file:
        evaluation = json.load(json_file)
    with open(os.path.join(dir, 'parameters', 'run.json')) as json_file:
        run = json.load(json_file)
    with open(os.path.join(dir, 'parameters', 'environment.json')) as json_file:
        env = json.load(json_file)
    with open(os.path.join(dir, 'parameters', 'design.json')) as json_file:
        design = json.load(json_file)
    
    for name, value in in_hp.items():
        hp[name] = value
    for name, value in in_evaluation.items():
        evaluation[name] = value
    for name, value in in_run.items():
        run[name] = value
    
    hp = namedtuple('hp', hp.keys())(**hp)
    evaluation = namedtuple('evaluation', evaluation.keys())(**evaluation)
    run = namedtuple('run', run.keys())(**run)
    env = namedtuple('env', env.keys())(**env)
    design = namedtuple('design', design.keys())(**design)

    return hp, evaluation, run, env, design
