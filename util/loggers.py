import os
import csv
import pandas as pd
import numpy as np
from typing import Dict, Any
from util.utils import object2dict

useless_args = ['device', 'writer','start_time', 'projection_size', 'log_dir', 'train', 'eval']
useless_args_train = ['data_dir', 'm_backbone', 'm_update', 'save_model', 'num_workers', 'temperature', 'n_proj']
useless_args_train = ['data_dir', 'm_backbone', 'm_update', 'save_model', 'num_workers']

class CsvLogger:
    def __init__(self, args) -> None:
        self.acc = 0.0
        self.dataset = args.train.dataset.name
        self.model = args.train.model
        self.output_dir = args.train.save_dir
        self.experiment_id = args.exp
        self.log_dir = args.log_dir
        self.blond_male = 0.0
        self.nonblonde_male = 0.0
        self.blond_female = 0.0
        self.nonblond_female = 0.0

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        self.acc = mean_acc

    def log_celeb(self, blond_male, nonblonde_male, blond_female, nonblond_female) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        self.blond_male = blond_male
        self.nonblonde_male = nonblonde_male
        self.blond_female = blond_female
        self.nonblond_female = nonblond_female

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        new_args = {}
        base_args = vars(args)
        extra_args = object2dict(args)

        columns = []
        for key, val in extra_args.items():
            if key not in useless_args:
                columns.append(key)
                new_args[key] = val

        for key, val in extra_args['train'].items():
            if isinstance(val, list):
                columns.append(key)
                new_args[key] = ''.join(map(str, val))
            elif isinstance(val, dict):
                for key1, val1 in val.items():
                    new_key = key + '_' + key1
                    if key1 not in useless_args_train:
                        columns.append(new_key)
                        new_args[new_key] = val1
            else:
                if key not in useless_args_train:
                    columns.append(key)
                    new_args[key] = val


        columns.append('acc')
        new_args['acc'] = self.acc

        results_dir = os.path.join(self.log_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        write_headers = False
        path = os.path.join(results_dir, "results.csv")

        # df = pd.DataFrame(extra_args)
        # df.to_csv(path)

        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(new_args)
