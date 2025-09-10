from abc import ABC, abstractmethod
import torch
from argparse import Namespace
import json
from itertools import product
import random
import sqlite3
import os

from .parse_config import parse_parameter_config

class HPSearch(ABC):
    def __init__(self, root, combinations):
        self.root = root
        self.combination = None
        self.path = None
        self.chkpt = None

        if not os.path.isdir(root):
            os.mkdir(root)

        con = sqlite3.connect(os.path.join(self.root, "registry.db"))
        cur = con.cursor()

        try:
            self.init_db(combinations, cur)
        except Exception as e:
            print('search already initialized, continuing where we left off...', e)

        con.commit()
        cur.close()

        self.get_new_combination()

    @classmethod
    def from_namespace(cls, root, search_space, args, k=None):
        arg_list = tuple([vars(args)[key] for key in search_space])
        if k is not None:
            arg_product = random.choices(list(product(*arg_list)), k=k)
        else:
            arg_product = list(product(*arg_list))

        combinations = []

        for args in arg_product:
            combinations.append({key: value for key, value in zip(search_space, args)})

        return cls(root, combinations)
    
    @classmethod
    def from_config(cls, root, config, k=None):
        combinations = parse_parameter_config(config)
        
        if k is not None:
            combinations = random.choices(combinations, k=k)

        return cls(root, combinations)
    
    @abstractmethod
    def init_db(self, combinations, cur):
        raise NotImplementedError('init_db is not implemented as part of the HPSearch abstract class')

    def update_namespace(self, args=None):

        try:
            combination = json.loads(self.combination)
        except TypeError as e:
            print("no combination left to load...")
            return None
        
        args = vars(args) if args is not None else dict()
        args.update(combination)
        return Namespace(**args)

    def save_checkpoint(self, states):
        with open(self.path, "wb") as fp:
            torch.save(states, fp)

    @abstractmethod
    def finish_combination(self, metric_value):
        raise NotImplementedError('finish_combination is not implemented as part of the HPSearch abstract class')

    @abstractmethod
    def get_new_combination(self, metric_value=None):
        raise NotImplementedError('get_new_combination is not implemented as part of the HPSearch abstract class')

    @abstractmethod
    def get_path(self, combination):
        raise NotImplementedError('get_path is not implemented as part of the HPSearch abstract class')
